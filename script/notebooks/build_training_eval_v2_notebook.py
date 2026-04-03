from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_training_eval.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 03 — Training / Evaluation Phase Closure v2\n\n"
            "Notebook này khép lại pha training/evaluation của nhánh `codex/preprocessing-v4-core`.\n\n"
            "Workflow active gồm 5 phần:\n\n"
            "1. **Kiểm tra input contract** trên feature table `v2`.\n"
            "2. **Clean benchmark** để chọn candidate theo `val_auc`, calibration và threshold lock trên `val`.\n"
            "3. **Model-level nuisance audit** với `AUC_nat` trên real-only `4:4:4 vs 4:2:0`.\n"
            "4. **Degradation suite** với `AUC_xdeg` trên các phép hậu kỳ bắt buộc.\n"
            "5. **Family ablation + phase closure summary** để chốt branch nào còn đáng giữ cho vòng tiếp theo.\n\n"
            "Artifact được lưu dưới `audit_output/validation/<run_name>/` theo các pha:\n"
            "- `phase1_clean_benchmark`\n"
            "- `phase2_model_nuisance`\n"
            "- `phase3_degradation_suite`\n"
            "- `phase4_family_ablation`\n"
            "- `phase5_phase_closure`\n\n"
            "Notebook này **không** còn nhúng `feature-audit` dựa trên metadata cũ; nó chỉ orchestration benchmark và audit ở mức model theo active spec."
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
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.training import (\n"
            "    ABLATION_FEATURE_SET_COLUMNS,\n"
            "    DEGRADATION_SPECS,\n"
            "    FEATURE_SET_COLUMNS,\n"
            "    load_training_table,\n"
            "    run_training_phase_closure,\n"
            ")\n\n"
            "from src.feature_extraction import ALL_FEATURE_KEYS\n\n"
            "FEATURE_TABLE = Path(os.getenv('TRAINING_V2_FEATURE_TABLE', PROJECT_ROOT / 'features' / 'feature_extraction_v2_rgb248_exact.csv'))\n"
            "DEFAULT_RUN_NAME = f\"training_v2_phase_closure_{datetime.now().strftime('%Y%m%d')}\"\n"
            "RUN_NAME = os.getenv('TRAINING_V2_RUN_NAME', DEFAULT_RUN_NAME)\n"
            "OUTPUT_ROOT = PROJECT_ROOT / 'audit_output' / 'validation' / RUN_NAME\n"
            "PHASE1_DIR = OUTPUT_ROOT / 'phase1_clean_benchmark'\n"
            "PHASE2_DIR = OUTPUT_ROOT / 'phase2_model_nuisance'\n"
            "PHASE3_DIR = OUTPUT_ROOT / 'phase3_degradation_suite'\n"
            "PHASE4_DIR = OUTPUT_ROOT / 'phase4_family_ablation'\n"
            "PHASE5_DIR = OUTPUT_ROOT / 'phase5_phase_closure'\n"
            "MODEL_OUTPUT_DIR = PROJECT_ROOT / 'models' / 'param' / RUN_NAME\n"
            "SUMMARY_PATH = OUTPUT_ROOT / 'summary.json'\n"
            "FORCE_RERUN = os.getenv('TRAINING_V2_FORCE_RERUN', '0') == '1'\n"
            "WORKERS = int(os.getenv('TRAINING_V2_WORKERS', '1'))\n"
            "SHOW_PROGRESS = os.getenv('TRAINING_V2_SHOW_PROGRESS', '1') == '1'\n\n"
            "def read_csv_if_exists(path: Path) -> pd.DataFrame:\n"
            "    if not path.exists():\n"
            "        return pd.DataFrame({'missing_file': [str(path)]})\n"
            "    return pd.read_csv(path)\n\n"
            "def read_json_if_exists(path: Path) -> dict:\n"
            "    if not path.exists():\n"
            "        return {'missing_file': str(path)}\n"
            "    return json.loads(path.read_text(encoding='utf-8'))\n\n"
            "print({\n"
            "    'feature_table': str(FEATURE_TABLE),\n"
            "    'run_name': RUN_NAME,\n"
            "    'output_root': str(OUTPUT_ROOT),\n"
            "    'model_output_dir': str(MODEL_OUTPUT_DIR),\n"
            "    'force_rerun': FORCE_RERUN,\n"
            "    'workers': WORKERS,\n"
            "    'show_progress': SHOW_PROGRESS,\n"
            "    'clean_feature_sets': list(FEATURE_SET_COLUMNS),\n"
            "    'ablation_feature_sets': list(ABLATION_FEATURE_SET_COLUMNS),\n"
            "    'degradation_suite': [spec.name for spec in DEGRADATION_SPECS],\n"
            "})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0. Kiểm tra input contract\n\n"
            "Cell này chỉ đọc feature table active và xác nhận contract trước khi chạy phase closure."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "training_frame = load_training_table(FEATURE_TABLE)\n"
            "{\n"
            "    'rows': int(len(training_frame)),\n"
            "    'feature_version': str(training_frame['feature_version'].iloc[0]),\n"
            "    'preprocess_version': str(training_frame['preprocess_version'].iloc[0]),\n"
            "    'split_role_counts': training_frame['split_role'].value_counts().to_dict(),\n"
            "    'generator_counts': training_frame['generator'].value_counts().to_dict(),\n"
            "    'feature_column_count': int(len(ALL_FEATURE_KEYS)),\n"
            "}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0.1. Inventory branch benchmark\n\n"
            "Cell này nhắc lại inventory branch nào sẽ được benchmark clean và branch nào sẽ đi vào family ablation."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "{\n"
            "    'clean_feature_sets': {name: len(cols) for name, cols in FEATURE_SET_COLUMNS.items()},\n"
            "    'ablation_feature_sets': {name: len(cols) for name, cols in ABLATION_FEATURE_SET_COLUMNS.items()},\n"
            "}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Run Or Load Phase Closure\n\n"
            "Nếu artifact đã tồn tại và `TRAINING_V2_FORCE_RERUN=0`, notebook sẽ nạp lại kết quả.\n"
            "Nếu cần chạy lại toàn bộ clean benchmark, nuisance audit, degradation suite và family ablation, đặt `TRAINING_V2_FORCE_RERUN=1`."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if SUMMARY_PATH.exists() and not FORCE_RERUN:\n"
            "    summary = read_json_if_exists(SUMMARY_PATH)\n"
            "else:\n"
            "    summary = run_training_phase_closure(\n"
            "        FEATURE_TABLE,\n"
            "        output_root=OUTPUT_ROOT,\n"
            "        model_output_dir=MODEL_OUTPUT_DIR,\n"
            "        workers=WORKERS,\n"
            "        force_rerun=FORCE_RERUN,\n"
            "        show_progress=SHOW_PROGRESS,\n"
            "    )\n"
            "summary"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Clean benchmark trên validation"))
    cells.append(
        nbf.v4.new_code_cell(
            "candidate_metrics = read_csv_if_exists(PHASE1_DIR / 'candidate_val_metrics.csv')\n"
            "candidate_metrics.sort_values(['val_auc', 'val_brier'], ascending=[False, True]).head(12)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Selected model trên clean split"))
    cells.append(
        nbf.v4.new_code_cell(
            "selected_metrics = read_csv_if_exists(PHASE1_DIR / 'selected_model_metrics.csv')\n"
            "selected_metrics"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. OOD breakdown và clean CFA gate coverage"))
    cells.append(
        nbf.v4.new_code_cell(
            "ood_by_generator = read_csv_if_exists(PHASE1_DIR / 'selected_model_ood_by_generator.csv')\n"
            "clean_cfa_gate_coverage = read_csv_if_exists(PHASE1_DIR / 'clean_cfa_gate_coverage.csv')\n"
            "ood_by_generator, clean_cfa_gate_coverage"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Model-level AUC_nat"))
    cells.append(
        nbf.v4.new_code_cell(
            "auc_nat_metrics = read_csv_if_exists(PHASE2_DIR / 'model_level_auc_nat.csv')\n"
            "auc_nat_metrics.sort_values(['split', 'auc_nat_abs', 'candidate_name'], ascending=[True, False, True])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5.1. Hỗ trợ nhãn nuisance"))
    cells.append(
        nbf.v4.new_code_cell(
            "nuisance_label_summary = read_csv_if_exists(PHASE2_DIR / 'nuisance_label_summary.csv')\n"
            "nuisance_label_summary"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Degradation suite: metrics gộp"))
    cells.append(
        nbf.v4.new_code_cell(
            "degradation_metrics = read_csv_if_exists(PHASE3_DIR / 'degradation_metrics.csv')\n"
            "degradation_metrics.loc[degradation_metrics['split'] == 'pooled_eval'].sort_values(['candidate_name', 'auc'], ascending=[True, False])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6.1. Degradation gaps so với clean"))
    cells.append(
        nbf.v4.new_code_cell(
            "degradation_gap_summary = read_csv_if_exists(PHASE3_DIR / 'degradation_gap_summary.csv')\n"
            "degradation_gap_summary.loc[degradation_gap_summary['split'] == 'pooled_eval'].sort_values(['candidate_name', 'auc_gap'])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6.2. CFA gate coverage dưới degradation"))
    cells.append(
        nbf.v4.new_code_cell(
            "degradation_cfa_gate_coverage = read_csv_if_exists(PHASE3_DIR / 'degradation_cfa_gate_coverage.csv')\n"
            "degradation_cfa_gate_coverage.sort_values(['degradation_name', 'split_role', 'label'])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7. Family ablation trên clean split"))
    cells.append(
        nbf.v4.new_code_cell(
            "ablation_candidate_metrics = read_csv_if_exists(PHASE4_DIR / 'ablation_candidate_val_metrics.csv')\n"
            "ablation_candidate_metrics.sort_values(['val_auc', 'val_brier'], ascending=[False, True])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7.1. Clean pooled metrics của các branch ablation"))
    cells.append(
        nbf.v4.new_code_cell(
            "ablation_clean_metrics = read_csv_if_exists(PHASE4_DIR / 'ablation_clean_metrics.csv')\n"
            "ablation_clean_metrics.loc[ablation_clean_metrics['split'] == 'pooled_eval'].sort_values(['auc', 'brier'], ascending=[False, True])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 8. Branch closure summary"))
    cells.append(
        nbf.v4.new_code_cell(
            "branch_closure_summary = read_csv_if_exists(PHASE4_DIR / 'branch_closure_summary.csv')\n"
            "branch_closure_summary.sort_values(['clean_pooled_auc', 'mean_xdeg_auc', 'auc_nat_abs'], ascending=[False, False, True])"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 9. Phase closure manifest"))
    cells.append(
        nbf.v4.new_code_cell(
            "phase_closure_manifest = read_json_if_exists(PHASE5_DIR / 'phase_closure_summary.json')\n"
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
