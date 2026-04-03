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
            "# 03 — Training & Evaluation v2\n\n"
            "Notebook này benchmark baseline training trên feature table `v2_rgb248_exact`.\n\n"
            "- Input: `features/feature_extraction_v2_rgb248_exact.csv`\n"
            "- Candidate sets: `control_minimal`, `always_on`, `always_on_plus_cfa_raw`, `always_on_plus_cfa_gated`, `full_v2`\n"
            "- Models: `logreg`, `lightgbm`\n"
            "- Selection: theo `val_auc`, calibration bằng calibration split, threshold khóa trên `val` với target `FPR <= 5%`"
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
            "from src.training import FEATURE_SET_COLUMNS, run_training_baseline\n\n"
            "FEATURE_TABLE = PROJECT_ROOT / 'features' / 'feature_extraction_v2_rgb248_exact.csv'\n"
            "RUN_NAME = 'training_v2_baseline_20260403'\n"
            "OUTPUT_DIR = PROJECT_ROOT / 'audit_output' / 'validation' / RUN_NAME\n"
            "FORCE_RERUN = os.getenv('TRAINING_V2_FORCE_RERUN', '0') == '1'\n"
            "SUMMARY_PATH = OUTPUT_DIR / 'summary.json'\n"
            "print({'feature_table': str(FEATURE_TABLE), 'output_dir': str(OUTPUT_DIR), 'feature_sets': list(FEATURE_SET_COLUMNS)})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Run or load baseline benchmark\n\n"
            "Nếu artifact đã tồn tại và `TRAINING_V2_FORCE_RERUN=False`, notebook chỉ load lại. Nếu không, notebook chạy toàn bộ benchmark."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if SUMMARY_PATH.exists() and not FORCE_RERUN:\n"
            "    summary = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))\n"
            "else:\n"
            "    summary = run_training_baseline(FEATURE_TABLE, output_dir=OUTPUT_DIR)\n"
            "summary"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Candidate ranking on validation split"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "candidate_metrics = pd.read_csv(OUTPUT_DIR / 'candidate_val_metrics.csv')\n"
            "candidate_metrics.sort_values(['val_auc', 'val_brier'], ascending=[False, True]).head(10)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Selected model metrics"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "selected_metrics = pd.read_csv(OUTPUT_DIR / 'selected_model_metrics.csv')\n"
            "selected_metrics"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. OOD breakdown by generator"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "ood_by_generator = pd.read_csv(OUTPUT_DIR / 'selected_model_ood_by_generator.csv')\n"
            "ood_by_generator"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Selected model importance"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "family_importance = pd.read_csv(OUTPUT_DIR / 'selected_model_family_importance.csv')\n"
            "feature_importance = pd.read_csv(OUTPUT_DIR / 'selected_model_feature_importance.csv')\n"
            "family_importance, feature_importance.head(15)"
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
