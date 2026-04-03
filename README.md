# AI Detector Image Repository

## Cấu trúc đề xuất

- `app/`: ứng dụng phục vụ demo/API.
- `audit_output/`: artifact sinh ra từ các đợt audit và validation, đã nhóm theo `data_audit/`, `validation/`, `studies/`.
- `data/`: dữ liệu raw, cleaned và processed.
- `deploy/`: mã phục vụ pipeline deploy/demo.
- `docs/`: toàn bộ tài liệu dự án.
- `features/`: dataset đặc trưng đã trích chọn.
- `inference/`: pipeline suy luận và tiện ích runtime.
- `models/`: artifact huấn luyện, tham số, checkpoint và metric.
- `notebooks/`: notebook EDA, preprocessing, feature extraction, training/eval.
- `script/`: script nghiên cứu, kiểm định và tái lập thí nghiệm.
- `src/`: mã nguồn lõi đã được nhóm theo mục đích.

## Cây mã nguồn

- `src/preprocessing/`: pipeline tiền xử lý canonical.
- `src/feature_extraction/`: 4 nhóm handcrafted features và worker trích xuất.
- `src/training/`: benchmark/training baseline, calibration, metric và selection logic cho stack v2.
- `src/visualization/`: hàm debug/plot bám theo từng nhóm feature.
- `src/dataset_tools/`: công cụ làm sạch dữ liệu, strip metadata, chuyển đổi định dạng.

## Cây script

- `script/audit/`: audit dữ liệu và metadata.
- `script/validation/`: script kiểm định giả thuyết/pipeline.
- `script/studies/`: study mở rộng feature space hoặc hướng nghiên cứu mới.
- `script/notebooks/`: builder và executor để tái sinh notebook theo pipeline mới.

## Cây tài liệu

- `docs/specs/`: đặc tả chuẩn cần bám khi triển khai.
- `docs/reports/`: báo cáo kiểm định, phản biện và cập nhật nghiên cứu.
- `docs/reference/legacy_text/`: các ghi chú/spec gốc dạng `.txt` và tài liệu nền.

## Tài liệu nên đọc trước

- `docs/specs/preprocessing_pipeline_standard_v4.md`: chuẩn preprocessing hiện hành theo hướng geometry-safe exact crop, support gate và fail-closed input contract.
- `docs/specs/feature_extraction_standard_v2.md`: source-of-truth hiện tại cho phase feature extraction; chốt inventory feature v2, taxonomy `always-on / conditional / research-only / drop`, và hướng `multi-branch + nonlinear fusion`.
- `docs/specs/feature_extraction_standard_v1.md`: spec active trước đó, giữ lại để đối chiếu quyết định.
- `docs/reports/shortcut_risk_validation.md`: báo cáo risk hợp nhất, gồm shortcut, compression history, giới hạn lý thuyết và phán quyết cuối cho preprocessing policy.
- `docs/reports/feature_space_update.md`: báo cáo hợp nhất về mở rộng feature space, với kết luận ưu tiên khối CFA trước.
- `docs/reports/feature_spec_v0_review.md`: review mới cho spec feature v0, gồm thực nghiệm JPEG history, patch aggregation, và khuyến nghị mở rộng feature family.
- `docs/reports/feature_spec_v2_validation.md`: validation mới nhất trên cùng tập ảnh qua pipeline cũ và mới, có thêm diagnostics cho SLA mapping, shift redundancy, control generalization, cross-noise pathology, và framing `multi-feature + conditional branches`.
- `docs/reports/training_v2_baseline_20260403.md`: benchmark training baseline đầu tiên trên full feature table v2; xác nhận có thể train baseline ngay nhưng chưa đủ bằng chứng để gọi selected model là champion-safe.

## Nguyên tắc tổ chức

- Runtime code giữ nguyên ở `src/`, `inference/`, `deploy/`; không trộn với tài liệu nghiên cứu.
- Tài liệu quyết định kỹ thuật phải nằm trong `docs/specs/` hoặc `docs/reports/`, không để ở root.
- Artifact số liệu phải nằm trong `audit_output/`, không gắn lẫn vào notebook hay markdown mô tả.
- Dữ liệu, feature tables, model artifacts và validation outputs là generated artifacts; README và ignore rules phải làm rõ đâu là source-of-truth, đâu là output sinh ra.
