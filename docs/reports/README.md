# Bản đồ báo cáo

`docs/reports/` chứa các báo cáo giải thích **vì sao** nhánh hiện tại được thiết kế như vậy.

## 1. Bộ báo cáo active nên đọc trước

- [branch_experiment_overview.md](branch_experiment_overview.md)
  - báo cáo tổng quan của toàn nhánh
  - phù hợp nhất cho người mới vào dự án
  - giải thích mục tiêu, timeline, trạng thái từng pha và các quyết định kỹ thuật lớn

- [preprocessing_validation_summary.md](preprocessing_validation_summary.md)
  - báo cáo hợp nhất cho pha tiền xử lý
  - giải thích vì sao `v4_exact` được chọn và vì sao các hướng như padding/resize/JPEG bottleneck/chroma canonicalization bị loại

- [feature_space_validation_summary.md](feature_space_validation_summary.md)
  - báo cáo hợp nhất cho pha feature
  - giải thích vì sao stack 33 feature cũ không đủ, vì sao phải chuyển sang kiến trúc nhiều nhánh, và cơ sở của taxonomy hiện tại

- [training_baseline_validation.md](training_baseline_validation.md)
  - báo cáo hợp nhất cho training baseline
  - giải thích selected baseline hiện tại mạnh ở đâu, nguy hiểm ở đâu, và vì sao chưa được gọi là champion-safe

## 2. Các báo cáo versioned / historical support

Các file dưới đây vẫn được giữ lại vì giá trị lịch sử, đối chiếu và chứng minh chi tiết hơn theo từng vòng nghiên cứu:

- [shortcut_risk_validation.md](shortcut_risk_validation.md)
- [feature_space_update.md](feature_space_update.md)
- [feature_spec_v0_review.md](feature_spec_v0_review.md)
- [feature_spec_v1_validation.md](feature_spec_v1_validation.md)
- [feature_spec_v2_validation.md](feature_spec_v2_validation.md)
- [training_v2_baseline_20260403.md](training_v2_baseline_20260403.md)

## 3. Thứ tự đọc khuyến nghị

Người mới nên đọc:

1. [branch_experiment_overview.md](branch_experiment_overview.md)
2. [preprocessing_validation_summary.md](preprocessing_validation_summary.md)
3. [feature_space_validation_summary.md](feature_space_validation_summary.md)
4. [training_baseline_validation.md](training_baseline_validation.md)

Khi cần truy vết chi tiết của từng vòng phản biện/validation, mới quay sang nhóm `historical support`.

## 4. Nguyên tắc quản trị

- report active phải trả lời câu hỏi “vì sao thiết kế hiện tại được chọn”
- report historical giữ vai trò bằng chứng chi tiết và lịch sử tranh luận
- artifact số liệu không nằm trong `docs/reports/`; chúng nằm ở `audit_output/`
