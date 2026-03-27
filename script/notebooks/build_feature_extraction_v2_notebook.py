from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_feature_extraction.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 02 — Feature Extraction v2\n\n"
            "Notebook này chạy pha feature extraction theo `feature_extraction_standard_v2.md`.\n\n"
            "- Input: `data/processed_v4_rgb248_r4_exact/manifest.csv`\n"
            "- Feature families: `always-on`, `conditional CFA`, `research-only`\n"
            "- Core rule: notebook chỉ orchestration; toàn bộ logic trích xuất nằm trong `src/feature_extraction`."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.feature_extraction import (\n"
            "    ALL_FEATURE_KEYS,\n"
            "    DEFAULT_CONFIG,\n"
            "    load_feature_manifest,\n"
            "    results_to_frame,\n"
            "    run_feature_pipeline,\n"
            "    save_feature_table,\n"
            "    summarise_feature_table,\n"
            ")\n\n"
            "MANIFEST_PATH = PROJECT_ROOT / 'data' / 'processed_v4_rgb248_r4_exact' / 'manifest.csv'\n"
            "MAX_FILES_ENV = os.getenv('FEATURE_EXTRACT_MAX_FILES', '').strip()\n"
            "MAX_FILES = None if not MAX_FILES_ENV else int(MAX_FILES_ENV)\n"
            "WORKERS = int(os.getenv('FEATURE_EXTRACT_WORKERS', str(min(8, os.cpu_count() or 4))))\n"
            "FORCE_RERUN = os.getenv('FEATURE_EXTRACT_FORCE_RERUN', '0') == '1'\n"
            "SHOW_PROGRESS = os.getenv('FEATURE_EXTRACT_SHOW_PROGRESS', '0') == '1'\n"
            "RUN_NAME = 'feature_extraction_v2_rgb248_exact' if MAX_FILES is None else f'feature_extraction_v2_rgb248_exact_smoke_{MAX_FILES}'\n"
            "OUTPUT_CSV = PROJECT_ROOT / 'features' / (f'{RUN_NAME}.csv')\n"
            "AUDIT_ROOT = PROJECT_ROOT / 'audit_output' / 'validation' / RUN_NAME\n"
            "SUMMARY_PATH = AUDIT_ROOT / 'feature_extraction_summary.json'\n"
            "AUDIT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "CONFIG = DEFAULT_CONFIG\n\n"
            "print({'manifest': str(MANIFEST_PATH), 'max_files': MAX_FILES, 'workers': WORKERS, 'output_csv': str(OUTPUT_CSV), 'feature_version': CONFIG.feature_version})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Load accepted preprocessing manifest\n\n"
            "Cell này chỉ đọc manifest preprocessing v4, lọc `ACCEPTED`, gán `split_role`, và tùy chọn lấy sample smoke theo `FEATURE_EXTRACT_MAX_FILES`."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "manifest = load_feature_manifest(MANIFEST_PATH, config=CONFIG, max_files=MAX_FILES)\n"
            "manifest[['generator', 'label', 'split_role', 'patch_path']].head(10)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Run or load feature extraction\n\n"
            "Nếu file output đã tồn tại và `FORCE_RERUN=False`, notebook sẽ load lại. Nếu không, notebook sẽ chạy full extraction bằng API package."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if OUTPUT_CSV.exists() and not FORCE_RERUN:\n"
            "    feature_frame = pd.read_csv(OUTPUT_CSV)\n"
            "else:\n"
            "    results = run_feature_pipeline(\n"
            "        manifest,\n"
            "        config=CONFIG,\n"
            "        workers=WORKERS,\n"
            "        chunksize=32,\n"
            "        show_progress=SHOW_PROGRESS,\n"
            "    )\n"
            "    feature_frame = results_to_frame(results, config=CONFIG)\n"
            "    save_feature_table(feature_frame, OUTPUT_CSV)\n"
            "summary = summarise_feature_table(feature_frame, config=CONFIG)\n"
            "summary.update({'run_name': RUN_NAME, 'output_csv': str(OUTPUT_CSV), 'max_files': MAX_FILES, 'workers': WORKERS})\n"
            "SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')\n"
            "summary"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Status and split QA\n\n"
            "Kiểm tra nhanh trạng thái extraction, số hàng theo split, và shape output."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "feature_frame.groupby(['split_role', 'status']).size().unstack(fill_value=0)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Feature preview\n\n"
            "Xem một số cột quan trọng của nhánh `always-on` và `conditional`."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "preview_cols = [\n"
            "    'generator', 'label', 'split_role', 'status',\n"
            "    'frs_mid_variance', 'fft_mid_logenergy', 'spatial_snr_ratio',\n"
            "    'cfa_rg_pi_xy', 'cfa_bg_pi_xy', 'cfa_validity_score'\n"
            "]\n"
            "feature_frame[preview_cols].head(12)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Conditional CFA validity summary\n\n"
            "Cell này chỉ xem phân bố `cfa_validity_score` để phục vụ bước audit/gating phía sau."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "feature_frame['cfa_validity_score'].describe()"
        )
    )

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    }
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding='utf-8')
    print(f'Wrote {NOTEBOOK_PATH}')


if __name__ == '__main__':
    main()
