from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_eda_visualization.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 04 - Audit-Driven EDA (v4_exact + v2)\n\n"
            "Notebook 4 duoc thiet ke theo 3 phase, chi doc du lieu audit/artifact da sinh ra, khong tao du lieu gia:\n\n"
            "1. EDA sau tien xu ly (v4_exact)\n"
            "2. EDA sau trich chon dac trung (v2)\n"
            "3. EDA artifact eval mo hinh sau train va kiem tra feature\n\n"
            "Nguyen tac:\n"
            "- Moi ket luan phai truy duoc den file audit cu the\n"
            "- Phase 2 overlap/correlation chi dung split train_core de tranh leakage\n"
            "- Neu artifact khong ton tai, notebook se bao ro va skip phase, khong bịa so lieu"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n\n"
            "from datetime import datetime\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "plt.rcParams['figure.dpi'] = 130\n"
            "plt.rcParams['axes.titlesize'] = 12\n"
            "plt.rcParams['axes.labelsize'] = 10\n"
            "plt.rcParams['legend.fontsize'] = 9\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed_v4_rgb248_r4_exact'\n"
            "MANIFEST_PATH = PROCESSED_ROOT / 'manifest.csv'\n"
            "SNAPSHOT_SUMMARY_PATH = PROJECT_ROOT / 'audit_output' / 'validation' / 'spec_v4_20260319' / 'snapshot_summary.json'\n"
            "GEOMETRY_FRONTIER_PATH = PROJECT_ROOT / 'audit_output' / 'validation' / 'spec_v4_20260319' / 'geometry_frontier_mult8.csv'\n"
            "RESIDUE_SCAN_PATH = PROJECT_ROOT / 'audit_output' / 'validation' / 'spec_v4_20260319' / 'geometry_residue_scan_crop_248.csv'\n\n"
            "FEATURE_TABLE_PATH = PROJECT_ROOT / 'features' / 'feature_extraction_v2_rgb248_exact.csv'\n"
            "FEATURE_STUDY_DIR = PROJECT_ROOT / 'audit_output' / 'studies' / 'feature_spec_v2_validation_20260325'\n"
            "FEATURE_SET_METRICS_PATH = FEATURE_STUDY_DIR / 'feature_set_metrics.csv'\n"
            "FEATURE_STUDY_SUMMARY_PATH = FEATURE_STUDY_DIR / 'summary.json'\n"
            "CONTROL_CORR_PATH = FEATURE_STUDY_DIR / 'diagnostics' / 'control_minimal_correlation_matrix.csv'\n\n"
            "TRAIN_AUDIT_DIR = PROJECT_ROOT / 'audit_output' / 'validation' / 'training_v2_phase_closure_20260403'\n"
            "PHASE1_DIR = TRAIN_AUDIT_DIR / 'phase1_clean_benchmark'\n"
            "PHASE2_DIR = TRAIN_AUDIT_DIR / 'phase2_model_nuisance'\n"
            "PHASE3_DIR = TRAIN_AUDIT_DIR / 'phase3_degradation_suite'\n"
            "PHASE4_DIR = TRAIN_AUDIT_DIR / 'phase4_family_ablation'\n"
            "PHASE5_DIR = TRAIN_AUDIT_DIR / 'phase5_phase_closure'\n\n"
            "CANDIDATE_METRICS_PATH = PHASE1_DIR / 'candidate_val_metrics.csv'\n"
            "SELECTED_METRICS_PATH = PHASE1_DIR / 'selected_model_metrics.csv'\n"
            "OOD_BY_GENERATOR_PATH = PHASE1_DIR / 'selected_model_ood_by_generator.csv'\n"
            "CLEAN_CFA_GATE_COVERAGE_PATH = PHASE1_DIR / 'clean_cfa_gate_coverage.csv'\n"
            "FEATURE_IMPORTANCE_PATH = PHASE1_DIR / 'selected_model_feature_importance.csv'\n"
            "FAMILY_IMPORTANCE_PATH = PHASE1_DIR / 'selected_model_family_importance.csv'\n\n"
            "AUC_NAT_METRICS_PATH = PHASE2_DIR / 'model_level_auc_nat.csv'\n"
            "DEGRADATION_GAP_PATH = PHASE3_DIR / 'degradation_gap_summary.csv'\n"
            "BRANCH_CLOSURE_PATH = PHASE4_DIR / 'branch_closure_summary.csv'\n"
            "PHASE_CLOSURE_JSON = PHASE5_DIR / 'phase_closure_summary.json'\n\n"
            "MODEL_PARAM_DIR = PROJECT_ROOT / 'models' / 'param'\n"
            "RUN_NAME = os.getenv('EDA4_RUN_NAME', f\"eda_visualization_v4_{datetime.now().strftime('%Y%m%d')}\")\n"
            "VIS_ROOT = PROJECT_ROOT / 'audit_output' / 'validation' / RUN_NAME\n"
            "PHASE1_PLOT_DIR = VIS_ROOT / 'phase1_preprocess'\n"
            "PHASE2_PLOT_DIR = VIS_ROOT / 'phase2_features'\n"
            "PHASE3_PLOT_DIR = VIS_ROOT / 'phase3_model'\n"
            "for path in [PHASE1_PLOT_DIR, PHASE2_PLOT_DIR, PHASE3_PLOT_DIR]:\n"
            "    path.mkdir(parents=True, exist_ok=True)\n\n"
            "def load_csv(path: Path, *, required: bool = True) -> pd.DataFrame | None:\n"
            "    if not path.exists():\n"
            "        if required:\n"
            "            raise FileNotFoundError(f'Missing required CSV: {path}')\n"
            "        print(f'[SKIP] Missing optional CSV: {path}')\n"
            "        return None\n"
            "    return pd.read_csv(path)\n\n"
            "def load_json(path: Path, *, required: bool = True) -> dict:\n"
            "    if not path.exists():\n"
            "        if required:\n"
            "            raise FileNotFoundError(f'Missing required JSON: {path}')\n"
            "        print(f'[SKIP] Missing optional JSON: {path}')\n"
            "        return {}\n"
            "    return json.loads(path.read_text(encoding='utf-8'))\n\n"
            "def overlap_coefficient(x0: np.ndarray, x1: np.ndarray, bins: int = 60) -> float:\n"
            "    x = np.concatenate([x0, x1])\n"
            "    if x.size == 0:\n"
            "        return float('nan')\n"
            "    if np.allclose(np.nanmin(x), np.nanmax(x)):\n"
            "        return 1.0\n"
            "    hist0, edges = np.histogram(x0, bins=bins, range=(np.nanmin(x), np.nanmax(x)), density=True)\n"
            "    hist1, _ = np.histogram(x1, bins=bins, range=(np.nanmin(x), np.nanmax(x)), density=True)\n"
            "    width = np.diff(edges)\n"
            "    return float(np.sum(np.minimum(hist0, hist1) * width))\n\n"
            "print({\n"
            "    'run_name': RUN_NAME,\n"
            "    'manifest': str(MANIFEST_PATH),\n"
            "    'feature_table': str(FEATURE_TABLE_PATH),\n"
            "    'training_audit_dir': str(TRAIN_AUDIT_DIR),\n"
            "    'vis_root': str(VIS_ROOT),\n"
            "})\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Phase 1 - EDA Sau Tien Xu Ly\n\n"
            "Phase nay dung manifest preprocess v4 va artifact geometry audit de kiem tra: yield theo nhan, support gate, crop frontier va residue choice."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "manifest = load_csv(MANIFEST_PATH, required=True)\n"
            "manifest['support'] = pd.to_numeric(manifest['support'], errors='coerce')\n"
            "snapshot_summary = load_json(SNAPSHOT_SUMMARY_PATH, required=False)\n"
            "geometry_frontier = load_csv(GEOMETRY_FRONTIER_PATH, required=False)\n"
            "residue_scan = load_csv(RESIDUE_SCAN_PATH, required=False)\n"
            "status_by_label = manifest.groupby(['label', 'status']).size().unstack(fill_value=0)\n"
            "pre_summary = {\n"
            "    'n_total': int(len(manifest)),\n"
            "    'n_accepted': int((manifest['status'] == 'ACCEPTED').sum()),\n"
            "    'status_counts': manifest['status'].value_counts().to_dict(),\n"
            "    'status_by_label': status_by_label.to_dict(),\n"
            "    'snapshot_summary': snapshot_summary,\n"
            "}\n"
            "pre_summary"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "status_order = ['ACCEPTED', 'LOW_SUPPORT', 'UNSUPPORTED_INPUT', 'DECODE_ERROR']\n"
            "status_colors = {'ACCEPTED': '#2a9d8f', 'LOW_SUPPORT': '#f4a261', 'UNSUPPORTED_INPUT': '#e76f51', 'DECODE_ERROR': '#264653'}\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "table = status_by_label.reindex(columns=status_order, fill_value=0)\n"
            "left = np.zeros(len(table), dtype=float)\n"
            "y = np.arange(len(table))\n"
            "for status in status_order:\n"
            "    values = table[status].to_numpy(dtype=float)\n"
            "    ax.barh(y, values, left=left, color=status_colors[status], label=status)\n"
            "    left += values\n"
            "ax.set_yticks(y)\n"
            "ax.set_yticklabels(table.index.tolist())\n"
            "ax.set_xlabel('Image count')\n"
            "ax.set_title('Preprocessing status by label')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE1_PLOT_DIR / 'status_by_label.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "for label, color in [('nature', '#457b9d'), ('ai', '#e63946')]:\n"
            "    values = manifest.loc[manifest['label'] == label, 'support'].dropna().to_numpy(dtype=float)\n"
            "    if values.size:\n"
            "        ax.hist(values, bins=40, alpha=0.5, label=label, color=color)\n"
            "ax.axvline(252, color='black', linestyle='--', linewidth=1.5, label='threshold=252')\n"
            "ax.set_xlabel('Support = min(height, width)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('Support distribution before exact crop gate')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE1_PLOT_DIR / 'support_distribution.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "gen_counts = manifest.groupby(['generator', 'status']).size().unstack(fill_value=0)\n"
            "gen_counts['total'] = gen_counts.sum(axis=1)\n"
            "gen_counts['accepted_rate'] = gen_counts.get('ACCEPTED', 0.0) / gen_counts['total'].replace(0, np.nan)\n"
            "top_gen = gen_counts.sort_values('total', ascending=False).head(12).reset_index()\n"
            "fig, ax = plt.subplots(figsize=(12, 5))\n"
            "ax.bar(top_gen['generator'], top_gen['accepted_rate'], color='#2a9d8f', alpha=0.9)\n"
            "ax.set_ylim(0.0, 1.0)\n"
            "ax.set_ylabel('Accepted rate')\n"
            "ax.set_title('Top generators by accepted rate (preprocessing v4)')\n"
            "ax.tick_params(axis='x', rotation=35)\n"
            "for i, row in top_gen.iterrows():\n"
            "    ax.text(i, min(0.98, row['accepted_rate'] + 0.015), f\"n={int(row['total'])}\", ha='center', va='bottom', fontsize=8)\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE1_PLOT_DIR / 'accepted_rate_by_generator.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if geometry_frontier is None:\n"
            "    print('[SKIP] geometry_frontier_mult8.csv not found')\n"
            "else:\n"
            "    geometry_frontier['crop_size'] = pd.to_numeric(geometry_frontier['crop_size'], errors='coerce')\n"
            "    fig, ax = plt.subplots(figsize=(9, 5))\n"
            "    ax.plot(geometry_frontier['crop_size'], geometry_frontier['accepted_ai'], marker='o', label='accepted_ai')\n"
            "    ax.plot(geometry_frontier['crop_size'], geometry_frontier['accepted_real'], marker='o', label='accepted_real')\n"
            "    ax.axvline(248, color='black', linestyle='--', linewidth=1.2, label='champion crop=248')\n"
            "    ax.set_xlabel('Crop size')\n"
            "    ax.set_ylabel('Acceptance rate')\n"
            "    ax.set_title('Geometry frontier (mult8 search space)')\n"
            "    ax.set_ylim(0.5, 1.01)\n"
            "    ax.legend(loc='best')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(PHASE1_PLOT_DIR / 'geometry_frontier_acceptance.png', dpi=170)\n"
            "    plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if residue_scan is None:\n"
            "    print('[SKIP] geometry_residue_scan_crop_248.csv not found')\n"
            "else:\n"
            "    residue_scan = residue_scan.sort_values('residue')\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "    ax.plot(residue_scan['residue'], residue_scan['mean_center_linf'], marker='o', color='#1d3557')\n"
            "    ax.axvline(4, color='black', linestyle='--', linewidth=1.2, label='champion residue=4')\n"
            "    ax.set_xlabel('Residue')\n"
            "    ax.set_ylabel('Mean center L_inf drift')\n"
            "    ax.set_title('Residue scan at crop=248')\n"
            "    ax.legend(loc='best')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(PHASE1_PLOT_DIR / 'residue_scan_crop248.png', dpi=170)\n"
            "    plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Phase 2 - EDA Sau Trich Chon Dac Trung\n\n"
            "Phase nay dung feature table v2 va study artifact de xem chat luong feature, overlap theo nhan, tuong quan va trade-off utility/robustness.\n"
            "Luu y: missingness asymmetry duoc show ro de tranh an branch-shortcut do dropna."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "features = load_csv(FEATURE_TABLE_PATH, required=True)\n"
            "feature_set_metrics = load_csv(FEATURE_SET_METRICS_PATH, required=False)\n"
            "feature_study_summary = load_json(FEATURE_STUDY_SUMMARY_PATH, required=False)\n"
            "control_corr = load_csv(CONTROL_CORR_PATH, required=False)\n"
            "feature_importance = load_csv(FEATURE_IMPORTANCE_PATH, required=False)\n"
            "features = features.loc[features['status'].astype(str).str.lower().eq('ok')].copy()\n"
            "meta_cols = {'source_file_path', 'patch_path', 'generator', 'label', 'split_role', 'dataset_name', 'preprocess_version', 'feature_version', 'status', 'error'}\n"
            "numeric_feature_cols = [c for c in features.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(features[c])]\n"
            "phase2_summary = {\n"
            "    'n_rows_ok': int(len(features)),\n"
            "    'n_numeric_features': int(len(numeric_feature_cols)),\n"
            "    'split_counts': features['split_role'].value_counts().to_dict(),\n"
            "    'label_counts': features['label'].value_counts().to_dict(),\n"
            "    'feature_study_samples': feature_study_summary.get('samples', {}),\n"
            "}\n"
            "phase2_summary"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "train_for_missing = features.loc[features['split_role'].eq('train_core')].copy()\n"
            "missing_rows = []\n"
            "for c in numeric_feature_cols:\n"
            "    ai_rate = pd.to_numeric(train_for_missing.loc[train_for_missing['label'].eq('ai'), c], errors='coerce').isna().mean()\n"
            "    nat_rate = pd.to_numeric(train_for_missing.loc[train_for_missing['label'].eq('nature'), c], errors='coerce').isna().mean()\n"
            "    missing_rows.append({'feature': c, 'missing_ai': float(ai_rate), 'missing_nature': float(nat_rate), 'abs_gap': float(abs(ai_rate - nat_rate))})\n"
            "missing_by_label = pd.DataFrame(missing_rows).sort_values('abs_gap', ascending=False)\n"
            "top_gap = missing_by_label.head(20).iloc[::-1]\n"
            "y = np.arange(len(top_gap))\n"
            "fig, ax = plt.subplots(figsize=(11, 7))\n"
            "ax.barh(y - 0.18, top_gap['missing_nature'].to_numpy(), height=0.35, label='nature', color='#457b9d')\n"
            "ax.barh(y + 0.18, top_gap['missing_ai'].to_numpy(), height=0.35, label='ai', color='#e63946')\n"
            "ax.set_yticks(y)\n"
            "ax.set_yticklabels(top_gap['feature'].tolist())\n"
            "ax.set_xlabel('Missing rate (train_core)')\n"
            "ax.set_title('Top 20 missingness asymmetry by label')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE2_PLOT_DIR / 'feature_missing_asymmetry_top20.png', dpi=170)\n"
            "plt.show()\n"
            "missing_by_label.head(20)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "train_df = features.loc[features['split_role'].eq('train_core')].copy()\n"
            "if train_df.empty:\n"
            "    raise RuntimeError('No train_core rows found in feature table')\n"
            "if feature_importance is not None and 'feature' in feature_importance.columns:\n"
            "    ranked = [f for f in feature_importance['feature'].astype(str).tolist() if f in numeric_feature_cols]\n"
            "    top_features = ranked[:9]\n"
            "else:\n"
            "    top_features = numeric_feature_cols[:9]\n"
            "fig, axes = plt.subplots(3, 3, figsize=(15, 12))\n"
            "axes = axes.ravel()\n"
            "overlap_rows = []\n"
            "for idx, feature_name in enumerate(top_features):\n"
            "    ax = axes[idx]\n"
            "    ai_series = pd.to_numeric(train_df.loc[train_df['label'].eq('ai'), feature_name], errors='coerce')\n"
            "    nat_series = pd.to_numeric(train_df.loc[train_df['label'].eq('nature'), feature_name], errors='coerce')\n"
            "    ai_missing = float(ai_series.isna().mean())\n"
            "    nat_missing = float(nat_series.isna().mean())\n"
            "    missing_gap = float(abs(ai_missing - nat_missing))\n"
            "    ai_vals = ai_series.dropna().to_numpy()\n"
            "    nat_vals = nat_series.dropna().to_numpy()\n"
            "    if ai_vals.size < 20 or nat_vals.size < 20:\n"
            "        ax.text(0.5, 0.5, f'{feature_name}\\ninsufficient data', ha='center', va='center')\n"
            "        ax.set_axis_off()\n"
            "        overlap_rows.append({\n"
            "            'feature': feature_name,\n"
            "            'overlap_coeff': float('nan'),\n"
            "            'missing_ai': ai_missing,\n"
            "            'missing_nature': nat_missing,\n"
            "            'missing_gap': missing_gap,\n"
            "            'n_ai_non_missing': int(ai_vals.size),\n"
            "            'n_nature_non_missing': int(nat_vals.size),\n"
            "            'missingness_proxy_flag': bool(missing_gap >= 0.02),\n"
            "        })\n"
            "        continue\n"
            "    combined = np.concatenate([ai_vals, nat_vals])\n"
            "    q1, q99 = np.nanpercentile(combined, [1, 99])\n"
            "    ai_clip = np.clip(ai_vals, q1, q99)\n"
            "    nat_clip = np.clip(nat_vals, q1, q99)\n"
            "    ovl_continuous = overlap_coefficient(nat_clip, ai_clip, bins=60)\n"
            "    ovl_missing_lower = min(1.0 - ai_missing, 1.0 - nat_missing) * ovl_continuous\n"
            "    ovl_missing_upper = min(ai_missing, nat_missing) + ovl_missing_lower\n"
            "    overlap_rows.append({\n"
            "        'feature': feature_name,\n"
            "        'overlap_coeff_continuous': ovl_continuous,\n"
            "        'overlap_coeff_missing_lower': ovl_missing_lower,\n"
            "        'overlap_coeff_missing_upper': ovl_missing_upper,\n"
            "        'missing_ai': ai_missing,\n"
            "        'missing_nature': nat_missing,\n"
            "        'missing_gap': missing_gap,\n"
            "        'n_ai_non_missing': int(ai_vals.size),\n"
            "        'n_nature_non_missing': int(nat_vals.size),\n"
            "        'missingness_proxy_flag': bool(missing_gap >= 0.02),\n"
            "    })\n"
            "    ax.hist(nat_clip, bins=60, density=True, alpha=0.45, color='#457b9d', label='nature')\n"
            "    ax.hist(ai_clip, bins=60, density=True, alpha=0.45, color='#e63946', label='ai')\n"
            "    ax.set_title(f'{feature_name} | ovl={ovl_continuous:.3f} | miss_ovl=[{ovl_missing_lower:.3f},{ovl_missing_upper:.3f}] | mgap={missing_gap:.3f}')\n"
            "    if idx == 0:\n"
            "        ax.legend(loc='best')\n"
            "for j in range(len(top_features), len(axes)):\n"
            "    axes[j].set_axis_off()\n"
            "fig.suptitle('Feature distribution overlap by label (train_core, clipped 1-99%)', y=1.02)\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE2_PLOT_DIR / 'feature_overlap_top9.png', dpi=170)\n"
            "plt.show()\n"
            "overlap_df = pd.DataFrame(overlap_rows)\n"
            "overlap_df.sort_values(['missingness_proxy_flag', 'missing_gap', 'overlap_coeff_missing_lower'], ascending=[False, False, True])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "target = train_df['label'].map({'nature': 0, 'ai': 1}).astype(float)\n"
            "feature_label_corr = []\n"
            "for c in numeric_feature_cols:\n"
            "    s = pd.to_numeric(train_df[c], errors='coerce')\n"
            "    corr = s.corr(target)\n"
            "    ai_missing = float(s.loc[train_df['label'].eq('ai')].isna().mean())\n"
            "    nat_missing = float(s.loc[train_df['label'].eq('nature')].isna().mean())\n"
            "    missing_gap = float(abs(ai_missing - nat_missing))\n"
            "    corr_abs = float(abs(corr)) if pd.notna(corr) else 0.0\n"
            "    ranking_score = float(max(corr_abs, missing_gap))\n"
            "    feature_label_corr.append((c, float(corr) if pd.notna(corr) else float('nan'), missing_gap, ranking_score))\n"
            "top_ranked = sorted(feature_label_corr, key=lambda x: x[3], reverse=True)[:20]\n"
            "top_corr_features = [name for name, _, _, _ in top_ranked]\n"
            "pd.DataFrame(top_ranked, columns=['feature', 'corr_with_label', 'missing_gap', 'ranking_score'])\n"
            "corr_input = train_df[top_corr_features].apply(pd.to_numeric, errors='coerce')\n"
            "complete_case = corr_input.dropna(axis=0, how='any')\n"
            "complete_ratio = float(len(complete_case) / len(corr_input)) if len(corr_input) > 0 else float('nan')\n"
            "if complete_ratio >= 0.9 and len(complete_case) >= 500:\n"
            "    corr_base = complete_case\n"
            "    corr_mode = 'complete_case'\n"
            "else:\n"
            "    corr_base = corr_input.copy()\n"
            "    labels = train_df['label']\n"
            "    for col in top_corr_features:\n"
            "        global_med = corr_base[col].median()\n"
            "        for label_name in ['ai', 'nature']:\n"
            "            label_mask = labels.eq(label_name)\n"
            "            label_med = corr_base.loc[label_mask, col].median()\n"
            "            fill_val = label_med if pd.notna(label_med) else global_med\n"
            "            corr_base.loc[label_mask & corr_base[col].isna(), col] = fill_val\n"
            "        corr_base[col] = corr_base[col].fillna(global_med)\n"
            "    corr_mode = 'label_conditional_median_imputed_due_to_low_complete_ratio'\n"
            "    print('[WARN] Complete-case coverage < 90%; using label-conditional median fallback to reduce survivorship bias.')\n"
            "corr_mat = corr_base.corr()\n"
            "print({'corr_mode': corr_mode, 'corr_rows': int(len(corr_base)), 'corr_row_ratio': complete_ratio})\n"
            "fig, ax = plt.subplots(figsize=(12, 10))\n"
            "im = ax.imshow(corr_mat.to_numpy(), cmap='coolwarm', vmin=-1, vmax=1)\n"
            "ax.set_xticks(np.arange(len(top_corr_features)))\n"
            "ax.set_xticklabels(top_corr_features, rotation=75, ha='right', fontsize=8)\n"
            "ax.set_yticks(np.arange(len(top_corr_features)))\n"
            "ax.set_yticklabels(top_corr_features, fontsize=8)\n"
            "ax.set_title(f'Feature correlation heatmap (train_core, top20 |corr(label)|, mode={corr_mode})')\n"
            "fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE2_PLOT_DIR / 'feature_correlation_heatmap_top20.png', dpi=170)\n"
            "plt.show()\n"
            "notna = corr_input.notna().astype(np.int32)\n"
            "pair_n = notna.T.dot(notna)\n"
            "pairs = []\n"
            "for i, fi in enumerate(top_corr_features):\n"
            "    for j in range(i + 1, len(top_corr_features)):\n"
            "        fj = top_corr_features[j]\n"
            "        val = corr_mat.iloc[i, j]\n"
            "        if pd.notna(val) and abs(val) >= 0.8:\n"
            "            pairs.append({'feature_a': fi, 'feature_b': fj, 'corr': float(val), 'n_pair_non_missing': int(pair_n.loc[fi, fj])})\n"
            "pd.DataFrame(pairs).sort_values('corr', key=lambda s: np.abs(s), ascending=False).head(20) if pairs else pd.DataFrame({'note': ['No |corr| >= 0.8 in top20']})"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if feature_set_metrics is None:\n"
            "    print('[SKIP] feature_set_metrics.csv not found')\n"
            "else:\n"
            "    v4 = feature_set_metrics.loc[feature_set_metrics['preprocess_version'].eq('v4_exact')].copy()\n"
            "    clean = v4.loc[v4['task'].eq('label_logo_clean'), ['feature_set', 'auc']].rename(columns={'auc': 'clean_auc'})\n"
            "    xdeg = v4.loc[v4['task'].astype(str).str.startswith('xdeg_'), ['feature_set', 'auc']].groupby('feature_set', as_index=False)['auc'].min().rename(columns={'auc': 'worst_xdeg_auc'})\n"
            "    nuis = v4.loc[v4['task'].eq('real_jpeg_444_vs_420'), ['feature_set', 'auc']].rename(columns={'auc': 'nuisance_auc'})\n"
            "    comp = clean.merge(xdeg, on='feature_set', how='left').merge(nuis, on='feature_set', how='left')\n"
            "    fig, ax = plt.subplots(figsize=(9, 6))\n"
            "    sc = ax.scatter(comp['clean_auc'], comp['worst_xdeg_auc'], c=comp['nuisance_auc'], cmap='viridis', s=70, alpha=0.9)\n"
            "    for _, row in comp.iterrows():\n"
            "        ax.annotate(row['feature_set'], (row['clean_auc'], row['worst_xdeg_auc']), fontsize=8, xytext=(4, 3), textcoords='offset points')\n"
            "    ax.set_xlabel('Clean AUC (label_logo_clean)')\n"
            "    ax.set_ylabel('Worst Cross-Degradation AUC')\n"
            "    ax.set_title('Feature-set utility vs robustness (v4_exact)')\n"
            "    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)\n"
            "    cb.set_label('Natural nuisance AUC (real 4:4:4 vs 4:2:0)')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(PHASE2_PLOT_DIR / 'feature_set_utility_vs_robustness.png', dpi=170)\n"
            "    plt.show()\n"
            "    comp.sort_values('clean_auc', ascending=False).head(20)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Phase 3 - EDA Artifact Eval Mo Hinh Sau Train + Kiem Tra Feature\n\n"
            "Phase nay doc artifact training baseline de xem ranking candidate, split metrics, OOD, gate coverage, score distribution va importance.\n"
            "Luu y: CFA gate la artifact cua feature pipeline dieu kien, khong phai preprocess v4_exact contract."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "candidate_metrics = load_csv(CANDIDATE_METRICS_PATH, required=True)\n"
            "selected_metrics = load_csv(SELECTED_METRICS_PATH, required=True)\n"
            "ood_by_generator = load_csv(OOD_BY_GENERATOR_PATH, required=False)\n"
            "clean_cfa_gate = load_csv(CLEAN_CFA_GATE_COVERAGE_PATH, required=False)\n"
            "feat_imp = load_csv(FEATURE_IMPORTANCE_PATH, required=False)\n"
            "fam_imp = load_csv(FAMILY_IMPORTANCE_PATH, required=False)\n"
            "auc_nat = load_csv(AUC_NAT_METRICS_PATH, required=False)\n"
            "degrad_gap = load_csv(DEGRADATION_GAP_PATH, required=False)\n"
            "branch_close = load_csv(BRANCH_CLOSURE_PATH, required=False)\n"
            "phase_closure_manifest = load_json(PHASE_CLOSURE_JSON, required=False)\n"
            "model_manifest = load_json(MODEL_PARAM_DIR / 'model_manifest.json', required=False)\n"
            "phase3_summary = {\n"
            "    'candidate_count': int(len(candidate_metrics)),\n"
            "    'selected_rows': int(len(selected_metrics)),\n"
            "    'has_ood_by_generator': ood_by_generator is not None,\n"
            "    'has_cfa_gate': clean_cfa_gate is not None,\n"
            "    'has_auc_nat': auc_nat is not None,\n"
            "    'has_degradation_gaps': degrad_gap is not None,\n"
            "    'has_branch_closure': branch_close is not None,\n"
            "    'has_phase_closure_manifest': bool(phase_closure_manifest),\n"
            "    'has_model_manifest': bool(model_manifest),\n"
            "}\n"
            "phase3_summary"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "rank = candidate_metrics.sort_values('val_auc', ascending=False).head(10).copy()\n"
            "rank['candidate_short'] = rank['candidate_name'].astype(str).str.replace('__', '\\n', regex=False)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "axes[0].barh(rank['candidate_short'][::-1], rank['val_auc'][::-1], color='#457b9d')\n"
            "axes[0].set_xlabel('Validation AUC')\n"
            "axes[0].set_title('Top candidates by val_auc')\n"
            "split_metrics = selected_metrics[['split', 'auc', 'brier', 'ece']].copy()\n"
            "x = np.arange(len(split_metrics))\n"
            "w = 0.24\n"
            "axes[1].bar(x - w, split_metrics['auc'], width=w, label='AUC', color='#2a9d8f')\n"
            "axes[1].bar(x, split_metrics['brier'], width=w, label='Brier', color='#f4a261')\n"
            "axes[1].bar(x + w, split_metrics['ece'], width=w, label='ECE', color='#e76f51')\n"
            "axes[1].set_xticks(x)\n"
            "axes[1].set_xticklabels(split_metrics['split'])\n"
            "axes[1].set_title('Selected model metrics by split')\n"
            "axes[1].legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE3_PLOT_DIR / 'candidate_ranking_and_split_metrics.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "if ood_by_generator is None or ood_by_generator.empty:\n"
            "    axes[0].text(0.5, 0.5, 'Missing OOD by generator artifact', ha='center', va='center')\n"
            "    axes[0].set_axis_off()\n"
            "else:\n"
            "    ood = ood_by_generator.sort_values('auc', ascending=True)\n"
            "    axes[0].barh(ood['generator'], ood['auc'], color='#457b9d')\n"
            "    axes[0].set_xlim(0.5, 1.0)\n"
            "    axes[0].set_xlabel('AUC')\n"
            "    axes[0].set_title('OOD AUC by generator')\n"
            "if clean_cfa_gate is None or clean_cfa_gate.empty:\n"
            "    axes[1].text(0.5, 0.5, 'Missing CFA gate coverage artifact', ha='center', va='center')\n"
            "    axes[1].set_axis_off()\n"
            "else:\n"
            "    pivot = clean_cfa_gate.pivot(index='split_role', columns='label', values='gate_rate').sort_index()\n"
            "    im = axes[1].imshow(pivot.to_numpy(), cmap='YlGnBu', vmin=0.0, vmax=1.0)\n"
            "    axes[1].set_xticks(np.arange(pivot.shape[1]))\n"
            "    axes[1].set_xticklabels(pivot.columns.tolist())\n"
            "    axes[1].set_yticks(np.arange(pivot.shape[0]))\n"
            "    axes[1].set_yticklabels(pivot.index.tolist())\n"
            "    axes[1].set_title('CFA gate rate artifact (conditional feature branch)')\n"
            "    for i in range(pivot.shape[0]):\n"
            "        for j in range(pivot.shape[1]):\n"
            "            v = pivot.iloc[i, j]\n"
            "            axes[1].text(j, i, f'{v:.3f}', ha='center', va='center', color='black', fontsize=8)\n"
            "    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)\n"
            "    print('[NOTE] CFA gate coverage is a model-feature artifact, not preprocessing v4_exact contract evidence.')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE3_PLOT_DIR / 'ood_and_cfa_gate.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "if auc_nat is None or auc_nat.empty:\n"
            "    axes[0].text(0.5, 0.5, 'Missing model-level AUC_nat artifact', ha='center', va='center')\n"
            "    axes[0].set_axis_off()\n"
            "else:\n"
            "    nat_df = auc_nat.sort_values('auc_nat_abs', ascending=False).head(15)\n"
            "    axes[0].barh(nat_df['candidate_name'][::-1], nat_df['auc_nat_abs'][::-1], color='#e76f51')\n"
            "    axes[0].set_xlabel('|AUC_nat - 0.5|')\n"
            "    axes[0].set_title('Nuisance sensitivity (|AUC_nat - 0.5|)')\n"
            "if degrad_gap is None or degrad_gap.empty:\n"
            "    axes[1].text(0.5, 0.5, 'Missing degradation gap artifact', ha='center', va='center')\n"
            "    axes[1].set_axis_off()\n"
            "else:\n"
            "    deg_pool = degrad_gap.loc[degrad_gap['split'].eq('pooled_eval')].sort_values('auc_gap', ascending=True).head(15)\n"
            "    axes[1].barh(deg_pool['candidate_name'][::-1], deg_pool['auc_gap'][::-1], color='#2a9d8f')\n"
            "    axes[1].set_xlabel('AUC gap (Clean vs Degradation)')\n"
            "    axes[1].set_title('Degradation robustness gap (pooled_eval)')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE3_PLOT_DIR / 'nuisance_and_degradation_gaps.png', dpi=170)\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "if feat_imp is None or feat_imp.empty:\n"
            "    axes[0].text(0.5, 0.5, 'Missing feature importance artifact', ha='center', va='center')\n"
            "    axes[0].set_axis_off()\n"
            "else:\n"
            "    top_feat = feat_imp.sort_values('importance', ascending=True).tail(15)\n"
            "    axes[0].barh(top_feat['feature'], top_feat['importance'], color='#1d3557')\n"
            "    axes[0].set_title('Top 15 feature importance')\n"
            "if branch_close is None or branch_close.empty:\n"
            "    axes[1].text(0.5, 0.5, 'Missing branch_closure_summary artifact', ha='center', va='center')\n"
            "    axes[1].set_axis_off()\n"
            "else:\n"
            "    b_pool = branch_close.sort_values('clean_pooled_auc', ascending=True).tail(10)\n"
            "    axes[1].barh(b_pool['candidate_name'][::-1], b_pool['clean_pooled_auc'][::-1], color='#457b9d')\n"
            "    axes[1].set_title('Branch Closure Clean Pooled AUC')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PHASE3_PLOT_DIR / 'importance_and_branch_closure.png', dpi=170)\n"
            "plt.show()\n"
            "phase_closure_manifest"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
