"""Feature-level shortcut audit helpers for training v2 workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.feature_extraction import ALL_FEATURE_KEYS

from .constants import FEATURE_FAMILY_MAP


def _normalize_path_series(series: pd.Series) -> pd.Series:
	return (
		series.astype(str)
		.str.replace("\\", "/", regex=False)
		.str.strip()
		.str.lower()
	)


def _as_bool(series: pd.Series) -> pd.Series:
	if pd.api.types.is_bool_dtype(series):
		return series.astype("boolean")
	lowered = series.astype(str).str.strip().str.lower()
	mapped = lowered.map(
		{
			"true": True,
			"1": True,
			"yes": True,
			"y": True,
			"false": False,
			"0": False,
			"no": False,
			"n": False,
			"nan": pd.NA,
			"none": pd.NA,
			"": pd.NA,
		}
	)
	return mapped.astype("boolean")


def _safe_auc_binary(target: pd.Series, values: pd.Series) -> float:
	target_num = pd.to_numeric(target, errors="coerce")
	values_num = pd.to_numeric(values, errors="coerce")
	finite = np.isfinite(values_num.to_numpy(dtype=np.float64, na_value=np.nan))
	mask = target_num.notna() & values_num.notna() & pd.Series(finite, index=values_num.index)
	if int(mask.sum()) < 4:
		return float("nan")
	y = target_num.loc[mask].astype(np.int32).to_numpy()
	x = values_num.loc[mask].astype(np.float64).to_numpy()
	if np.unique(y).size < 2:
		return float("nan")
	if np.unique(x).size < 2:
		return 0.5
	try:
		return float(roc_auc_score(y, x))
	except ValueError:
		return float("nan")


def _auc_abs(value: float) -> float:
	if np.isnan(value):
		return float("nan")
	return float(max(value, 1.0 - value))


def _nanmax(values: list[float]) -> float:
	finite = [float(v) for v in values if not np.isnan(v)]
	if not finite:
		return float("nan")
	return float(max(finite))


def _mutual_information_bits_binary(y: np.ndarray, z: np.ndarray) -> float:
	if y.size == 0 or z.size == 0 or y.size != z.size:
		return float("nan")
	eps = 1e-12
	out = 0.0
	for y_val in (0, 1):
		for z_val in (0, 1):
			joint = float(np.mean((y == y_val) & (z == z_val)))
			if joint <= eps:
				continue
			py = float(np.mean(y == y_val))
			pz = float(np.mean(z == z_val))
			if py <= eps or pz <= eps:
				continue
			out += joint * np.log2(joint / (py * pz))
	return float(out)


def _shortcut_risk_level(max_nuisance_auc_abs: float) -> str:
	if np.isnan(max_nuisance_auc_abs):
		return "unknown"
	if max_nuisance_auc_abs >= 0.8:
		return "high"
	if max_nuisance_auc_abs >= 0.65:
		return "medium"
	return "low"


def _label_conditional_rate(mask: pd.Series, labels: pd.Series, label_value: int) -> float:
	label_mask = labels.eq(label_value)
	valid_mask = mask.notna() & label_mask
	if int(valid_mask.sum()) == 0:
		return float("nan")
	return float(mask.loc[valid_mask].astype(bool).mean())


def attach_metadata_for_feature_audit(
	frame: pd.DataFrame,
	*,
	metadata_csv_path: Path | str,
) -> tuple[pd.DataFrame, float]:
	metadata_path = Path(metadata_csv_path)
	if not metadata_path.exists():
		raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

	usecols = {
		"file_path",
		"relative_path",
		"inferred_label",
		"inferred_generator",
		"format_detected",
		"image_mode",
		"has_alpha",
		"is_grayscale",
		"jpeg_subsampling",
	}
	metadata = pd.read_csv(
		metadata_path,
		usecols=lambda name: name in usecols,
		low_memory=False,
		encoding="utf-8-sig",
	)

	for column in sorted(usecols):
		if column not in metadata.columns:
			metadata[column] = pd.NA

	metadata = metadata.copy()
	metadata["file_path_norm"] = _normalize_path_series(metadata["file_path"])
	metadata = metadata.drop_duplicates(subset=["file_path_norm"], keep="first")

	audit = frame.copy()
	audit["source_file_norm"] = _normalize_path_series(audit["source_file_path"])
	merged = audit.merge(
		metadata,
		left_on="source_file_norm",
		right_on="file_path_norm",
		how="left",
		suffixes=("", "_meta"),
	)
	join_rate = float(merged["file_path"].notna().mean())

	if "has_alpha" in merged.columns:
		merged["has_alpha"] = _as_bool(merged["has_alpha"])
	if "is_grayscale" in merged.columns:
		merged["is_grayscale"] = _as_bool(merged["is_grayscale"])

	return merged, join_rate


def compute_shortcut_proxy_asymmetry(frame: pd.DataFrame) -> pd.DataFrame:
	labels = frame["y"].astype(np.int32)
	rows: list[dict[str, Any]] = []

	fmt_series = frame["format_detected"].astype(str).str.upper()
	fmt_proxy = pd.Series(pd.NA, index=frame.index, dtype="boolean")
	fmt_proxy.loc[fmt_series.eq("PNG")] = True
	fmt_proxy.loc[fmt_series.eq("JPEG")] = False

	mode_series = frame["image_mode"].astype(str).str.upper()
	mode_proxy = pd.Series(pd.NA, index=frame.index, dtype="boolean")
	mode_proxy.loc[mode_series.eq("RGBA")] = True
	mode_proxy.loc[mode_series.eq("RGB")] = False

	proxy_map: dict[str, pd.Series] = {
		"format_is_png": fmt_proxy,
		"mode_is_rgba": mode_proxy,
		"has_alpha": frame["has_alpha"],
		"is_grayscale": frame["is_grayscale"],
	}

	for proxy_name, proxy_mask in proxy_map.items():
		proxy_bool = _as_bool(proxy_mask)
		p_ai = _label_conditional_rate(proxy_bool, labels, 1)
		p_nature = _label_conditional_rate(proxy_bool, labels, 0)
		valid = proxy_bool.notna()
		if int(valid.sum()) == 0:
			mi_bits = float("nan")
		else:
			mi_bits = _mutual_information_bits_binary(
				labels.loc[valid].to_numpy(dtype=np.int32),
				proxy_bool.loc[valid].astype(np.int32).to_numpy(dtype=np.int32),
			)
		rows.append(
			{
				"proxy": proxy_name,
				"scope": "all",
				"p_proxy_given_ai": p_ai,
				"p_proxy_given_nature": p_nature,
				"abs_rate_gap": float(abs(p_ai - p_nature)) if not np.isnan(p_ai) and not np.isnan(p_nature) else float("nan"),
				"mi_bits": mi_bits,
				"n_valid": int(valid.sum()),
			}
		)

	real_only = frame["y"].eq(0)
	real_subsample = frame.loc[real_only, "jpeg_subsampling"].astype(str)
	real_valid = real_subsample.isin(["4:4:4", "4:2:0"])
	n_real_valid = int(real_valid.sum())
	p_420 = (
		float(real_subsample.loc[real_valid].eq("4:2:0").mean())
		if n_real_valid > 0
		else float("nan")
	)
	rows.append(
		{
			"proxy": "real_jpeg_subsampling_is_420",
			"scope": "nature_only",
			"p_proxy_given_ai": float("nan"),
			"p_proxy_given_nature": p_420,
			"abs_rate_gap": float("nan"),
			"mi_bits": float("nan"),
			"n_valid": n_real_valid,
		}
	)

	return pd.DataFrame(rows)


def _mean_abs_z_shift(
	values: pd.Series,
	*,
	train_mask: pd.Series,
	eval_mask: pd.Series,
) -> float:
	train_values = pd.to_numeric(values.loc[train_mask], errors="coerce")
	train_mean = float(train_values.mean())
	train_std = float(train_values.std(ddof=0))
	if np.isnan(train_mean) or np.isnan(train_std) or train_std <= 1e-12:
		return float("nan")
	eval_values = pd.to_numeric(values.loc[eval_mask], errors="coerce")
	z = (eval_values - train_mean) / train_std
	if z.empty:
		return float("nan")
	return float(np.nanmean(np.abs(z.to_numpy(dtype=np.float64, na_value=np.nan))))


def compute_feature_audit_metrics(frame: pd.DataFrame) -> pd.DataFrame:
	feature_keys = [name for name in ALL_FEATURE_KEYS if name in frame.columns]

	split_train = frame["split_role"].eq("train_core")
	split_val = frame["split_role"].eq("val")
	split_id = frame["split_role"].eq("id_test")
	split_ood = frame["split_role"].eq("ood_eval")
	ai_mask = frame["y"].eq(1)
	nature_mask = frame["y"].eq(0)

	sub_mask = frame["y"].eq(0) & frame["jpeg_subsampling"].astype(str).isin(["4:4:4", "4:2:0"])
	sub_target = frame.loc[sub_mask, "jpeg_subsampling"].astype(str).eq("4:2:0").astype(np.int32)

	fmt_series = frame["format_detected"].astype(str).str.upper()
	fmt_mask = fmt_series.isin(["JPEG", "PNG"])
	fmt_target = fmt_series.eq("PNG").astype(np.int32)

	alpha_series = _as_bool(frame["has_alpha"])
	alpha_mask = alpha_series.notna()
	alpha_target = alpha_series.astype("float").fillna(np.nan)

	gray_series = _as_bool(frame["is_grayscale"])
	gray_mask = gray_series.notna()
	gray_target = gray_series.astype("float").fillna(np.nan)

	mode_series = frame["image_mode"].astype(str).str.upper()
	mode_mask = mode_series.isin(["RGB", "RGBA"])
	mode_target = mode_series.eq("RGBA").astype(np.int32)

	rows: list[dict[str, Any]] = []
	for feature in feature_keys:
		values = pd.to_numeric(frame[feature], errors="coerce")

		label_auc = _safe_auc_binary(frame["y"], values)
		label_auc_abs = _auc_abs(label_auc)

		val_auc = _safe_auc_binary(frame.loc[split_val, "y"], values.loc[split_val])
		id_auc = _safe_auc_binary(frame.loc[split_id, "y"], values.loc[split_id])
		ood_auc = _safe_auc_binary(frame.loc[split_ood, "y"], values.loc[split_ood])

		nuisance_sub_auc = _safe_auc_binary(sub_target, values.loc[sub_mask])
		nuisance_sub_auc_abs = _auc_abs(nuisance_sub_auc)

		nuisance_fmt_ai_auc = _safe_auc_binary(
			fmt_target.loc[fmt_mask & ai_mask],
			values.loc[fmt_mask & ai_mask],
		)
		nuisance_fmt_ai_auc_abs = _auc_abs(nuisance_fmt_ai_auc)
		nuisance_fmt_nature_auc = _safe_auc_binary(
			fmt_target.loc[fmt_mask & nature_mask],
			values.loc[fmt_mask & nature_mask],
		)
		nuisance_fmt_nature_auc_abs = _auc_abs(nuisance_fmt_nature_auc)

		nuisance_alpha_ai_auc = _safe_auc_binary(
			alpha_target.loc[alpha_mask & ai_mask],
			values.loc[alpha_mask & ai_mask],
		)
		nuisance_alpha_ai_auc_abs = _auc_abs(nuisance_alpha_ai_auc)
		nuisance_alpha_nature_auc = _safe_auc_binary(
			alpha_target.loc[alpha_mask & nature_mask],
			values.loc[alpha_mask & nature_mask],
		)
		nuisance_alpha_nature_auc_abs = _auc_abs(nuisance_alpha_nature_auc)

		nuisance_mode_ai_auc = _safe_auc_binary(
			mode_target.loc[mode_mask & ai_mask],
			values.loc[mode_mask & ai_mask],
		)
		nuisance_mode_ai_auc_abs = _auc_abs(nuisance_mode_ai_auc)
		nuisance_mode_nature_auc = _safe_auc_binary(
			mode_target.loc[mode_mask & nature_mask],
			values.loc[mode_mask & nature_mask],
		)
		nuisance_mode_nature_auc_abs = _auc_abs(nuisance_mode_nature_auc)

		nuisance_gray_ai_auc = _safe_auc_binary(
			gray_target.loc[gray_mask & ai_mask],
			values.loc[gray_mask & ai_mask],
		)
		nuisance_gray_ai_auc_abs = _auc_abs(nuisance_gray_ai_auc)
		nuisance_gray_nature_auc = _safe_auc_binary(
			gray_target.loc[gray_mask & nature_mask],
			values.loc[gray_mask & nature_mask],
		)
		nuisance_gray_nature_auc_abs = _auc_abs(nuisance_gray_nature_auc)

		max_nuisance_auc_abs = _nanmax(
			[
				nuisance_sub_auc_abs,
				nuisance_fmt_ai_auc_abs,
				nuisance_fmt_nature_auc_abs,
				nuisance_alpha_ai_auc_abs,
				nuisance_alpha_nature_auc_abs,
				nuisance_mode_ai_auc_abs,
				nuisance_mode_nature_auc_abs,
				nuisance_gray_ai_auc_abs,
				nuisance_gray_nature_auc_abs,
			]
		)

		shift_val = _mean_abs_z_shift(values, train_mask=split_train, eval_mask=split_val)
		shift_id = _mean_abs_z_shift(values, train_mask=split_train, eval_mask=split_id)
		shift_ood = _mean_abs_z_shift(values, train_mask=split_train, eval_mask=split_ood)
		max_shift = _nanmax([shift_val, shift_id, shift_ood])

		if np.isnan(max_nuisance_auc_abs) or np.isnan(label_auc_abs) or label_auc_abs <= 1e-6:
			shortcut_to_signal_ratio = float("nan")
		else:
			shortcut_to_signal_ratio = float(max_nuisance_auc_abs / label_auc_abs)

		rows.append(
			{
				"feature": feature,
				"family": FEATURE_FAMILY_MAP.get(feature, "unknown"),
				"label_auc": label_auc,
				"label_auc_abs": label_auc_abs,
				"label_auc_val": val_auc,
				"label_auc_id_test": id_auc,
				"label_auc_ood_eval": ood_auc,
				"nuisance_auc_real_subsampling": nuisance_sub_auc,
				"nuisance_auc_real_subsampling_abs": nuisance_sub_auc_abs,
				"nuisance_auc_format_png_ai_only": nuisance_fmt_ai_auc,
				"nuisance_auc_format_png_ai_only_abs": nuisance_fmt_ai_auc_abs,
				"nuisance_auc_format_png_nature_only": nuisance_fmt_nature_auc,
				"nuisance_auc_format_png_nature_only_abs": nuisance_fmt_nature_auc_abs,
				"nuisance_auc_has_alpha_ai_only": nuisance_alpha_ai_auc,
				"nuisance_auc_has_alpha_ai_only_abs": nuisance_alpha_ai_auc_abs,
				"nuisance_auc_has_alpha_nature_only": nuisance_alpha_nature_auc,
				"nuisance_auc_has_alpha_nature_only_abs": nuisance_alpha_nature_auc_abs,
				"nuisance_auc_mode_rgba_ai_only": nuisance_mode_ai_auc,
				"nuisance_auc_mode_rgba_ai_only_abs": nuisance_mode_ai_auc_abs,
				"nuisance_auc_mode_rgba_nature_only": nuisance_mode_nature_auc,
				"nuisance_auc_mode_rgba_nature_only_abs": nuisance_mode_nature_auc_abs,
				"nuisance_auc_is_grayscale_ai_only": nuisance_gray_ai_auc,
				"nuisance_auc_is_grayscale_ai_only_abs": nuisance_gray_ai_auc_abs,
				"nuisance_auc_is_grayscale_nature_only": nuisance_gray_nature_auc,
				"nuisance_auc_is_grayscale_nature_only_abs": nuisance_gray_nature_auc_abs,
				"max_nuisance_auc_abs": max_nuisance_auc_abs,
				"shortcut_to_signal_ratio": shortcut_to_signal_ratio,
				"shortcut_signal_gap": (
					float(max_nuisance_auc_abs - label_auc_abs)
					if not np.isnan(max_nuisance_auc_abs) and not np.isnan(label_auc_abs)
					else float("nan")
				),
				"mean_abs_z_shift_val": shift_val,
				"mean_abs_z_shift_id_test": shift_id,
				"mean_abs_z_shift_ood_eval": shift_ood,
				"max_abs_z_shift": max_shift,
				"shortcut_risk_level": _shortcut_risk_level(max_nuisance_auc_abs),
			}
		)

	feature_frame = pd.DataFrame(rows)
	feature_frame = feature_frame.sort_values(
		["max_nuisance_auc_abs", "label_auc_abs"],
		ascending=[False, False],
		ignore_index=True,
	)
	return feature_frame


def compute_family_audit_metrics(feature_metrics: pd.DataFrame) -> pd.DataFrame:
	grouped = feature_metrics.groupby("family", as_index=False).agg(
		feature_count=("feature", "size"),
		label_auc_abs_mean=("label_auc_abs", "mean"),
		label_auc_abs_median=("label_auc_abs", "median"),
		label_auc_abs_max=("label_auc_abs", "max"),
		max_nuisance_auc_abs_mean=("max_nuisance_auc_abs", "mean"),
		max_nuisance_auc_abs_max=("max_nuisance_auc_abs", "max"),
		shortcut_to_signal_ratio_mean=("shortcut_to_signal_ratio", "mean"),
		max_abs_z_shift_mean=("max_abs_z_shift", "mean"),
		max_abs_z_shift_max=("max_abs_z_shift", "max"),
	)

	high_counts = (
		feature_metrics.assign(is_high=feature_metrics["shortcut_risk_level"].eq("high").astype(np.int32))
		.groupby("family", as_index=False)["is_high"]
		.sum()
		.rename(columns={"is_high": "high_risk_feature_count"})
	)
	medium_high_counts = (
		feature_metrics.assign(
			is_medium_or_high=feature_metrics["shortcut_risk_level"].isin(["medium", "high"]).astype(np.int32)
		)
		.groupby("family", as_index=False)["is_medium_or_high"]
		.sum()
		.rename(columns={"is_medium_or_high": "medium_or_high_risk_feature_count"})
	)

	grouped = grouped.merge(high_counts, on="family", how="left")
	grouped = grouped.merge(medium_high_counts, on="family", how="left")
	grouped["high_risk_feature_count"] = grouped["high_risk_feature_count"].fillna(0).astype(np.int32)
	grouped["medium_or_high_risk_feature_count"] = (
		grouped["medium_or_high_risk_feature_count"].fillna(0).astype(np.int32)
	)

	return grouped.sort_values("max_nuisance_auc_abs_max", ascending=False, ignore_index=True)


def save_feature_audit_artifacts(
	*,
	output_dir: Path | str,
	feature_metrics: pd.DataFrame,
	family_metrics: pd.DataFrame,
	proxy_asymmetry: pd.DataFrame,
) -> dict[str, str]:
	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)

	files = {
		"feature_metrics_csv": str((root / "feature_audit_metrics.csv").resolve()),
		"family_metrics_csv": str((root / "feature_family_audit_metrics.csv").resolve()),
		"proxy_asymmetry_csv": str((root / "shortcut_proxy_label_asymmetry.csv").resolve()),
		"top_shortcut_risk_csv": str((root / "feature_shortcut_risk_top20.csv").resolve()),
		"summary_json": str((root / "feature_audit_summary.json").resolve()),
	}

	feature_metrics.to_csv(root / "feature_audit_metrics.csv", index=False, encoding="utf-8-sig")
	family_metrics.to_csv(root / "feature_family_audit_metrics.csv", index=False, encoding="utf-8-sig")
	proxy_asymmetry.to_csv(root / "shortcut_proxy_label_asymmetry.csv", index=False, encoding="utf-8-sig")
	feature_metrics.head(20).to_csv(root / "feature_shortcut_risk_top20.csv", index=False, encoding="utf-8-sig")
	return files


def run_feature_shortcut_audit(
	frame: pd.DataFrame,
	*,
	feature_table_path: Path | str,
	metadata_csv_path: Path | str,
	output_dir: Path | str,
) -> dict[str, Any]:
	merged, join_rate = attach_metadata_for_feature_audit(frame, metadata_csv_path=metadata_csv_path)
	feature_metrics = compute_feature_audit_metrics(merged)
	family_metrics = compute_family_audit_metrics(feature_metrics)
	proxy_asymmetry = compute_shortcut_proxy_asymmetry(merged)

	risk_counts = (
		feature_metrics["shortcut_risk_level"]
		.value_counts(dropna=False)
		.to_dict()
	)
	summary: dict[str, Any] = {
		"feature_table_path": str(Path(feature_table_path).resolve()),
		"metadata_csv_path": str(Path(metadata_csv_path).resolve()),
		"rows": int(len(frame)),
		"feature_count": int(feature_metrics.shape[0]),
		"metadata_join_rate": join_rate,
		"high_risk_threshold": 0.8,
		"medium_risk_threshold": 0.65,
		"risk_level_counts": {str(k): int(v) for k, v in risk_counts.items()},
		"high_risk_feature_count": int(feature_metrics["shortcut_risk_level"].eq("high").sum()),
		"claim_scope": "empirical",
	}

	files = save_feature_audit_artifacts(
		output_dir=output_dir,
		feature_metrics=feature_metrics,
		family_metrics=family_metrics,
		proxy_asymmetry=proxy_asymmetry,
	)
	summary["files"] = files
	Path(files["summary_json"]).write_text(
		json.dumps(summary, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	return summary


__all__ = [
	"attach_metadata_for_feature_audit",
	"compute_family_audit_metrics",
	"compute_feature_audit_metrics",
	"compute_shortcut_proxy_asymmetry",
	"run_feature_shortcut_audit",
	"save_feature_audit_artifacts",
]
