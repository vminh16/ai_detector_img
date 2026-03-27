from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02b_eda_visualization.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# v4 Phase Visualization\n\n"
            "Notebook trực quan hóa được tách thành 3 phase tương ứng với pipeline mới:\n\n"
            "1. Tiền xử lý\n"
            "2. Trích chọn đặc trưng\n"
            "3. Phân tích mô hình\n\n"
            "Ở thời điểm hiện tại notebook sẽ render đầy đủ phase preprocessing từ output v4 mới. Hai phase sau sẽ tự kiểm tra artifact v4 và skip sạch nếu chưa có."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "from IPython.display import Markdown, display\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.visualization import (\n"
            "    render_feature_phase_report,\n"
            "    render_model_phase_report,\n"
            ")\n\n"
            "PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed_v4_rgb248_r4_exact'\n"
            "MANIFEST_PATH = PROCESSED_ROOT / 'manifest.csv'\n"
            "FEATURES_PATH = PROJECT_ROOT / 'features' / 'features_dataset_v4_rgb248_r4_exact.csv'\n"
            "MODEL_ROOT = PROJECT_ROOT / 'models' / '04_artifacts_v4_rgb248_r4_exact'\n"
            "VIS_ROOT = PROJECT_ROOT / 'audit_output' / 'validation' / 'spec_v4_20260319' / 'visualization_v4_rgb248_r4_exact'\n"
            "PRE_VIS = VIS_ROOT / 'preprocessing'\n"
            "FEATURE_VIS = VIS_ROOT / 'feature_extraction'\n"
            "MODEL_VIS = VIS_ROOT / 'model_analysis'\n\n"
            "PLOT_ROOT = PRE_VIS / 'plots'\n"
            "PLOT_ROOT.mkdir(parents=True, exist_ok=True)\n"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Phase 1 — Tiền xử lý"))
    cells.append(
        nbf.v4.new_code_cell(
            "manifest = pd.read_csv(MANIFEST_PATH)\n"
            "manifest['support'] = pd.to_numeric(manifest['support'], errors='coerce')\n"
            "status_by_label = manifest.groupby(['label', 'status']).size().unstack(fill_value=0)\n"
            "pre_summary = {\n"
            "    'n_total': int(len(manifest)),\n"
            "    'status_counts': manifest['status'].value_counts().to_dict(),\n"
            "    'status_by_label': status_by_label.to_dict(),\n"
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
            "fig.savefig(PLOT_ROOT / 'status_by_label.png', dpi=160)\n"
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
            "fig.savefig(PLOT_ROOT / 'support_distribution.png', dpi=160)\n"
            "plt.show()"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "gen_table = manifest.groupby(['generator', 'status']).size().unstack(fill_value=0).reindex(columns=status_order, fill_value=0)\n"
            "gen_table = gen_table.loc[gen_table.sum(axis=1).sort_values(ascending=False).head(12).index]\n"
            "fig, ax = plt.subplots(figsize=(12, 5))\n"
            "bottom = np.zeros(len(gen_table), dtype=float)\n"
            "x = np.arange(len(gen_table))\n"
            "for status in status_order:\n"
            "    values = gen_table[status].to_numpy(dtype=float)\n"
            "    ax.bar(x, values, bottom=bottom, color=status_colors[status], label=status)\n"
            "    bottom += values\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels(gen_table.index.tolist(), rotation=35, ha='right')\n"
            "ax.set_ylabel('Image count')\n"
            "ax.set_title('Top generators by preprocessing status')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PLOT_ROOT / 'status_by_generator.png', dpi=160)\n"
            "plt.show()"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "accepted = manifest.loc[(manifest['status'] == 'ACCEPTED') & (manifest['saved_patch'] == True)].copy()\n"
            "accepted[['crop_origin_x', 'crop_origin_y']] = accepted[['crop_origin_x', 'crop_origin_y']].apply(pd.to_numeric, errors='coerce')\n"
            "fig, ax = plt.subplots(figsize=(6, 6))\n"
            "ax.scatter(accepted['crop_origin_x'], accepted['crop_origin_y'], s=6, alpha=0.35, color='#1d3557', edgecolors='none')\n"
            "ax.set_xlabel('crop_origin_x')\n"
            "ax.set_ylabel('crop_origin_y')\n"
            "ax.set_title('Accepted crop origins')\n"
            "fig.tight_layout()\n"
            "fig.savefig(PLOT_ROOT / 'crop_origin_scatter.png', dpi=160)\n"
            "plt.show()"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "gallery_parts = []\n"
            "for mode in ['RGB', 'RGBA']:\n"
            "    subset = accepted.loc[accepted['input_mode'] == mode]\n"
            "    if not subset.empty:\n"
            "        gallery_parts.append(subset.sample(n=min(4, len(subset)), random_state=42))\n"
            "gallery = pd.concat(gallery_parts, ignore_index=True) if gallery_parts else accepted.sample(n=min(8, len(accepted)), random_state=42)\n"
            "n = len(gallery)\n"
            "cols = min(4, n)\n"
            "rows = int(np.ceil(n / cols))\n"
            "fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))\n"
            "axes = np.atleast_1d(axes).reshape(rows, cols)\n"
            "for ax in axes.flat:\n"
            "    ax.axis('off')\n"
            "for ax, (_, row) in zip(axes.flat, gallery.iterrows()):\n"
            "    patch = np.load(row['output_path'])\n"
            "    ax.imshow(patch)\n"
            "    ax.set_title(f\"{row['generator']} | {row['input_mode']}\")\n"
            "    ax.axis('off')\n"
            "fig.suptitle('Accepted patch gallery after preprocessing v4', fontsize=14)\n"
            "fig.tight_layout()\n"
            "fig.savefig(PLOT_ROOT / 'patch_gallery.png', dpi=160)\n"
            "plt.show()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Phase 2 — Trích chọn đặc trưng"))
    cells.append(
        nbf.v4.new_code_cell(
            "feature_summary = render_feature_phase_report(FEATURES_PATH, FEATURE_VIS)\n"
            "feature_summary"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Phase 3 — Phân tích mô hình"))
    cells.append(
        nbf.v4.new_code_cell(
            "model_summary = render_model_phase_report(MODEL_ROOT, MODEL_VIS)\n"
            "model_summary"
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
