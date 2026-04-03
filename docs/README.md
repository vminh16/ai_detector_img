# Bản đồ tài liệu của nhánh thực nghiệm

Thư mục `docs/` được tổ chức theo ba lớp:

- `specs/`
  - mô tả hệ thống **đang phải chạy như thế nào**
- `reports/`
  - giải thích **vì sao** các quyết định hiện tại được chọn
- `reference/`
  - tài liệu legacy và vật liệu lịch sử của dự án trước khi tái cấu trúc

## 1. Nếu chỉ đọc bốn tài liệu để hiểu toàn nhánh

1. [reports/active/branch_experiment_overview.md](reports/active/branch_experiment_overview.md)
2. [specs/active/preprocessing_standard.md](specs/active/preprocessing_standard.md)
3. [specs/active/feature_extraction_standard.md](specs/active/feature_extraction_standard.md)
4. [specs/active/training_evaluation_standard.md](specs/active/training_evaluation_standard.md)

Đây là bộ tài liệu ngắn nhất nhưng đủ để nắm:

- bài toán của dự án là gì,
- nhánh hiện tại đang ở đâu,
- vì sao từng pha được thiết kế như vậy,
- và bước tiếp theo là gì.

## 2. Đường đọc theo từng pha

### Tiền xử lý

- spec active: [specs/active/preprocessing_standard.md](specs/active/preprocessing_standard.md)
- report active: [reports/active/preprocessing_validation_summary.md](reports/active/preprocessing_validation_summary.md)
- tài liệu versioned hỗ trợ: [specs/archive/preprocessing_pipeline_standard_v4.md](specs/archive/preprocessing_pipeline_standard_v4.md)

### Trích chọn đặc trưng

- spec active: [specs/active/feature_extraction_standard.md](specs/active/feature_extraction_standard.md)
- report active: [reports/active/feature_space_validation_summary.md](reports/active/feature_space_validation_summary.md)
- tài liệu versioned hỗ trợ:
  - [specs/archive/feature_extraction_standard_v2.md](specs/archive/feature_extraction_standard_v2.md)
  - [reports/archive/feature_spec_v2_validation.md](reports/archive/feature_spec_v2_validation.md)

### Huấn luyện và đánh giá

- spec active: [specs/active/training_evaluation_standard.md](specs/active/training_evaluation_standard.md)
- report active: [reports/active/training_baseline_validation.md](reports/active/training_baseline_validation.md)
- tài liệu versioned hỗ trợ: [reports/archive/training_v2_baseline_20260403.md](reports/archive/training_v2_baseline_20260403.md)

## 3. Vai trò của thư mục `reference/`

`reference/` không còn là source-of-truth.

Nó chỉ giữ:

- spec/ghi chú cũ
- giả định ban đầu của pipeline legacy
- tài liệu nền dùng để đối chiếu các quyết định tái cấu trúc

Nếu cần triển khai hay đánh giá hệ hiện tại, không đọc `reference/` trước.
Hãy đọc `reports/` và `specs/` active trước.

## 4. Quy ước tài liệu trong nhánh này

- tài liệu active viết bằng tiếng Việt, có dấu đầy đủ
- mỗi tài liệu phải có vai trò rõ ràng, không gộp toàn bộ dự án vào một file duy nhất
- file versioned giữ lại để truy vết lịch sử quyết định
- artifact số liệu và notebook output không nằm trong `docs/`; chúng nằm ở `audit_output/`, `features/`, `models/`
