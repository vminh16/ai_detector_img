from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_preprocessing.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 01 — Preprocessing v4\n\n"
            "Notebook này chạy preprocessing champion v4 theo `spec_v4`.\n\n"
            "- Input: `data/raw`\n"
            "- Output mới: `data/processed_v4_rgb248_r4_exact`\n"
            "- Manifest: `data/processed_v4_rgb248_r4_exact/manifest.csv`\n"
            "- Audit run: `audit_output/validation/spec_v4_20260319/preprocessing_run_v4_rgb248_r4_exact`\n"
            "- Core contract: decode canonical, EXIF orientation, RGB/RGBA only, exact crop `248@4`, không pad/resize/JPEG bottleneck."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n\n"
            "import json\n"
            "import os\n"
            "import shutil\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.preprocessing import (\n"
            "    DEFAULT_CONFIG,\n"
            "    run_pipeline,\n"
            "    save_manifest,\n"
            "    scan_dataset_tree,\n"
            "    summarise_results,\n"
            ")\n\n"
            "RAW_ROOT = PROJECT_ROOT / 'data' / 'raw'\n"
            "PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed_v4_rgb248_r4_exact'\n"
            "MANIFEST_PATH = PROCESSED_ROOT / 'manifest.csv'\n"
            "RUN_AUDIT_ROOT = PROJECT_ROOT / 'audit_output' / 'validation' / 'spec_v4_20260319' / 'preprocessing_run_v4_rgb248_r4_exact'\n"
            "SUMMARY_PATH = RUN_AUDIT_ROOT / 'preprocessing_run_summary.json'\n"
            "WORKERS = min(12, os.cpu_count() or 8)\n"
            "FORCE_RERUN = False\n"
            "SHOW_PROGRESS = False\n"
            "CONFIG = DEFAULT_CONFIG\n"
            "RUN_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)\n\n"
            "print({'raw_root': str(RAW_ROOT), 'processed_root': str(PROCESSED_ROOT), 'manifest': str(MANIFEST_PATH), 'workers': WORKERS, 'preprocess_version': CONFIG.preprocess_version})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Scan raw dataset\n\n"
            "Quét cấu trúc dữ liệu đầu vào trước khi chạy pipeline để khóa số lượng và cấu trúc thư mục."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "dataset_scan = scan_dataset_tree(RAW_ROOT)\n"
            "{\n"
            "    'root': dataset_scan['root'],\n"
            "    'total_images': dataset_scan['total_images'],\n"
            "    'n_subdirs': len(dataset_scan['by_subdir']),\n"
            "    'sample_subdirs': dict(list(dataset_scan['by_subdir'].items())[:10]),\n"
            "}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Run preprocessing v4\n\n"
            "Nếu output mới chưa tồn tại, notebook sẽ chạy full preprocessing. Nếu manifest đã có và `FORCE_RERUN=False`, notebook chỉ load lại kết quả."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def manifest_summary(manifest: pd.DataFrame) -> dict[str, int]:\n"
            "    status_counts = manifest['status'].value_counts().to_dict()\n"
            "    return {\n"
            "        'total_scanned': int(len(manifest)),\n"
            "        'accepted': int(status_counts.get('ACCEPTED', 0)),\n"
            "        'low_support': int(status_counts.get('LOW_SUPPORT', 0)),\n"
            "        'unsupported_input': int(status_counts.get('UNSUPPORTED_INPUT', 0)),\n"
            "        'decode_error': int(status_counts.get('DECODE_ERROR', 0)),\n"
            "        'saved_patch': int(pd.to_numeric(manifest.get('saved_patch', pd.Series(dtype=int)), errors='coerce').fillna(0).astype(int).sum()),\n"
            "        'stale_output_removed': int(pd.to_numeric(manifest.get('stale_output_removed', pd.Series(dtype=int)), errors='coerce').fillna(0).astype(int).sum()),\n"
            "        'orientation_applied': int(pd.to_numeric(manifest.get('orientation_applied', pd.Series(dtype=int)), errors='coerce').fillna(0).astype(int).sum()),\n"
            "        'alpha_composited': int(pd.to_numeric(manifest.get('alpha_composited', pd.Series(dtype=int)), errors='coerce').fillna(0).astype(int).sum()),\n"
            "    }\n\n"
            "if FORCE_RERUN and PROCESSED_ROOT.exists():\n"
            "    shutil.rmtree(PROCESSED_ROOT)\n\n"
            "if MANIFEST_PATH.exists() and not FORCE_RERUN:\n"
            "    manifest = pd.read_csv(MANIFEST_PATH)\n"
            "    run_summary = manifest_summary(manifest)\n"
            "    run_summary['run_mode'] = 'load_existing_manifest'\n"
            "else:\n"
            "    results = run_pipeline(\n"
            "        RAW_ROOT,\n"
            "        PROCESSED_ROOT,\n"
            "        config=CONFIG,\n"
            "        workers=WORKERS,\n"
            "        overwrite=True,\n"
            "        show_progress=SHOW_PROGRESS,\n"
            "        log_failures=False,\n"
            "    )\n"
            "    save_manifest(results, MANIFEST_PATH)\n"
            "    manifest = pd.read_csv(MANIFEST_PATH)\n"
            "    run_summary = summarise_results(results)\n"
            "    run_summary['run_mode'] = 'fresh_pipeline_run'\n\n"
            "run_summary.update({\n"
            "    'raw_root': str(RAW_ROOT),\n"
            "    'processed_root': str(PROCESSED_ROOT),\n"
            "    'manifest_path': str(MANIFEST_PATH),\n"
            "    'workers': WORKERS,\n"
            "    'preprocess_version': CONFIG.preprocess_version,\n"
            "    'crop_size': CONFIG.crop_size,\n"
            "    'residue': [CONFIG.residue_x, CONFIG.residue_y],\n"
            "    'support_threshold': CONFIG.support_threshold,\n"
            "})\n"
            "SUMMARY_PATH.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding='utf-8')\n"
            "run_summary"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Post-run sanity checks\n\n"
            "Kiểm tra phân bố trạng thái, coverage theo label, mode input và vài hàng manifest đầu tiên."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "status_by_label = manifest.groupby(['label', 'status']).size().unstack(fill_value=0)\n"
            "status_by_label"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "manifest[['status', 'input_format', 'input_mode', 'normalized_mode', 'support', 'crop_origin_x', 'crop_origin_y', 'saved_patch']].head(12)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "accepted = manifest.loc[manifest['status'] == 'ACCEPTED'].copy()\n"
            "accepted[['label', 'generator', 'input_mode', 'output_path']].head(10)"
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
