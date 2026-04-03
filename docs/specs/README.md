# Bản đồ đặc tả

`docs/specs/` chứa các đặc tả active và historical của nhánh.

## 1. Bộ đặc tả active nên đọc trước

Đây là ba tài liệu source-of-truth hiện hành của nhánh:

- [preprocessing_standard.md](preprocessing_standard.md)
  - đặc tả tiền xử lý active
  - mô tả input/output contract, exact crop `248 @ 4`, status, metric và các phép bị cấm
  - phản ánh implementation hiện tại trong [src/preprocessing](C:/Users/USER/Desktop/ai_detector_img/src/preprocessing)

- [feature_extraction_standard.md](feature_extraction_standard.md)
  - đặc tả trích chọn đặc trưng active
  - mô tả taxonomy `always-on / conditional / research-only / drop`
  - phản ánh implementation hiện tại trong [src/feature_extraction](C:/Users/USER/Desktop/ai_detector_img/src/feature_extraction)

- [training_evaluation_standard.md](training_evaluation_standard.md)
  - đặc tả training/evaluation active
  - mô tả split contract, benchmark candidate, calibration, threshold lock, metric và điều kiện champion-readiness
  - phản ánh implementation hiện tại trong [src/training](C:/Users/USER/Desktop/ai_detector_img/src/training)

## 2. Các đặc tả versioned giữ lại để đối chiếu

Các file sau vẫn được giữ vì giá trị lịch sử và đối chiếu:

- [preprocessing_pipeline_standard_v4.md](preprocessing_pipeline_standard_v4.md)
  - bản versioned chi tiết của quyết định preprocessing v4
  - hiện đóng vai trò tài liệu nền/historical support cho `preprocessing_standard.md`

- [feature_extraction_standard_v2.md](feature_extraction_standard_v2.md)
  - bản versioned chi tiết của taxonomy feature v2
  - hiện đóng vai trò historical support cho `feature_extraction_standard.md`

- [feature_extraction_standard_v1.md](feature_extraction_standard_v1.md)
  - spec active trước đó, đã bị supersede

- [feature_extraction_standard_v0.md](feature_extraction_standard_v0.md)
  - bản nháp rất sớm, giữ để đối chiếu logic governance ban đầu

- [preprocessing_pipeline_standard.md](preprocessing_pipeline_standard.md)
  - tài liệu tiền xử lý cũ trước khi chốt v4
  - không còn là source-of-truth

## 3. Thứ tự đọc khuyến nghị

Nếu muốn hiểu nhanh hệ hiện tại:

1. [preprocessing_standard.md](preprocessing_standard.md)
2. [feature_extraction_standard.md](feature_extraction_standard.md)
3. [training_evaluation_standard.md](training_evaluation_standard.md)

Nếu cần truy vết lý do lịch sử:

4. [preprocessing_pipeline_standard_v4.md](preprocessing_pipeline_standard_v4.md)
5. [feature_extraction_standard_v2.md](feature_extraction_standard_v2.md)

## 4. Nguyên tắc quản trị

- tài liệu active phải trả lời câu hỏi “pha này hiện phải chạy như thế nào”
- tài liệu versioned giữ vai trò chứng cứ lịch sử và diễn tiến quyết định
- mọi spec active phải tham chiếu được tới report và artifact thực nghiệm tương ứng
