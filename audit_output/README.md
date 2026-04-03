# Audit Output Map

## Mục đích

`audit_output/` chỉ chứa artifact sinh ra từ:

- audit dữ liệu,
- validation giả thuyết preprocessing,
- study mở rộng feature,
- các đợt spec/proxy audit.

Không để notebook, báo cáo diễn giải hay code nguồn ở đây.

## Cấu trúc hiện tại

- `data_audit/`
  - `metadata/`: bảng per-file, report markdown, summary JSON, shortcut screening.
  - `dataset_profile/`: class/generator/OOD breakdown, histogram, format distribution, symmetry checklist.
  - `duplicates/`: duplicate report và duplicate summary.
- `validation/`
  - `pipeline_revision/`: artifact của bài kiểm định pipeline revision.
  - `risk_solution/`: artifact của bài kiểm định `RISK&SOLUTION`.
  - `spec_v4_20260319/`: artifact của đợt spec/proxy audit ngày `2026-03-19`.
    - root level: geometry, proxy, governance, snapshot artifact.
    - `implementation_checks/`: smoke test và integration check cho code v4.
    - `preprocessing_run_v4_rgb248_r4_exact/`: summary của run preprocessing v4 trên full raw snapshot.
    - `visualization_v4_rgb248_r4_exact/`: JSON trạng thái phase và thư mục `plots/` cho trực quan hóa.
- `studies/`
  - `update_feature/`: artifact của study mở rộng feature space.

## Quy ước

- File rất lớn dạng bảng gốc để trong `metadata/` hoặc đúng study/validation sinh ra nó.
- JSON tổng hợp, CSV metric và parquet feature phải đi cùng study/validation tương ứng.
- Nếu thêm audit mới, tạo một thư mục con theo tên study hoặc theo ngày thay vì đặt file trực tiếp ở root.
