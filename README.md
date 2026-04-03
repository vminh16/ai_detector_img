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

Nếu muốn hiểu nhanh toàn bộ nhánh thực nghiệm hiện tại, nên đọc theo thứ tự sau:

1. `docs/reports/branch_experiment_overview.md`
   - tổng quan nhánh, timeline, trạng thái từng pha và câu hỏi khoa học cốt lõi
2. `docs/specs/preprocessing_standard.md`
   - đặc tả tiền xử lý active
3. `docs/specs/feature_extraction_standard.md`
   - đặc tả feature extraction active
4. `docs/specs/training_evaluation_standard.md`
   - đặc tả training/evaluation active
5. `docs/reports/preprocessing_validation_summary.md`
   - báo cáo vì sao `v4_exact` được chọn
6. `docs/reports/feature_space_validation_summary.md`
   - báo cáo vì sao feature space phải chuyển sang kiến trúc nhiều nhánh
7. `docs/reports/training_baseline_validation.md`
   - báo cáo baseline training hiện tại và các rủi ro còn lại

## Nguyên tắc tổ chức

- Runtime code giữ nguyên ở `src/`, `inference/`, `deploy/`; không trộn với tài liệu nghiên cứu.
- Tài liệu quyết định kỹ thuật phải nằm trong `docs/specs/` hoặc `docs/reports/`, không để ở root.
- Artifact số liệu phải nằm trong `audit_output/`, không gắn lẫn vào notebook hay markdown mô tả.
- `docs/specs/` chứa source-of-truth active; các file versioned cũ được giữ lại như historical support.
- `docs/reports/` chứa cả bộ report active đã hợp nhất và các report historical để truy vết quyết định.
- Dữ liệu, feature tables, model artifacts và validation outputs là generated artifacts; README và ignore rules phải làm rõ đâu là source-of-truth, đâu là output sinh ra.
