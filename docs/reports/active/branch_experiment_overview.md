# Tổng quan nhánh thực nghiệm `codex/preprocessing-v4-core`

## 1. Mục đích của nhánh này

Nhánh này tồn tại để trả lời một câu hỏi rất cụ thể:

> Với dữ liệu ảnh web in-the-wild, nơi ảnh thật chủ yếu là JPEG còn ảnh AI chủ yếu là PNG,
> làm thế nào xây dựng một pipeline phát hiện ảnh do mô hình khuếch tán sinh ra mà không để
> mô hình chỉ học lịch sử nén, subsampling, mode ảnh hoặc các shortcut dữ liệu khác?

Đây không phải là một bài toán “chỉ cần chọn mô hình mạnh hơn”.
Vấn đề cốt lõi của dự án là **nhiễu tạp do pipeline và do dữ liệu**:

- ảnh thật và ảnh AI có lịch sử nén khác nhau ngay từ đầu,
- nhiều đặc trưng cổ điển trong pháp y ảnh phản ứng mạnh với JPEG/subsampling,
- một số metadata hoặc mode ảnh có thể trở thành shortcut nhị phân,
- nếu thiết kế tiền xử lý sai, mô hình có thể đạt AUC cao nhưng thực chất chỉ đang đọc codec history.

Vì vậy, nhánh này được xây dựng theo hướng:

1. **sửa đúng gốc vấn đề ở pha tiền xử lý**,
2. **đánh giá lại toàn bộ feature space dưới góc nhìn nuisance**,
3. **chỉ sau đó mới quay lại training**.

## 2. Toàn cảnh dự án sau khi tách nhánh

Pipeline hiện tại của nhánh gồm ba pha chính:

1. **Tiền xử lý**
   - chuẩn hiện hành: [preprocessing_standard.md](../../specs/active/preprocessing_standard.md)
   - thực thi ở [src/preprocessing](C:/Users/USER/Desktop/ai_detector_img/src/preprocessing)

2. **Trích chọn đặc trưng**
   - chuẩn hiện hành: [feature_extraction_standard.md](../../specs/active/feature_extraction_standard.md)
   - thực thi ở [src/feature_extraction](C:/Users/USER/Desktop/ai_detector_img/src/feature_extraction)

3. **Huấn luyện và đánh giá baseline**
   - chuẩn hiện hành: [training_evaluation_standard.md](../../specs/active/training_evaluation_standard.md)
   - thực thi ở [src/training](C:/Users/USER/Desktop/ai_detector_img/src/training)

Notebook tương ứng:

- [01_preprocessing.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/01_preprocessing.ipynb)
- [02_feature_extraction.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/02_feature_extraction.ipynb)
- [03_training_eval.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/03_training_eval.ipynb)

## 3. Trạng thái dữ liệu và artifact hiện tại

### 3.1. Raw snapshot

Snapshot raw hiện hành được khóa trong audit preprocessing v4:

- `87,971` file ảnh trên đĩa
- có `36` dòng metadata cũ bị stale trong parquet cũ

Điểm quan trọng:

- mọi script active của nhánh này đều đã chuyển sang lấy **live snapshot trên disk** làm source-of-truth,
- metadata cũ chỉ còn giá trị tham khảo lịch sử.

### 3.2. Kết quả tiền xử lý v4

Run canonical preprocessing v4 tạo ra:

- `85,615` patch `ACCEPTED`
- `1,573` ảnh `LOW_SUPPORT`
- `783` ảnh `UNSUPPORTED_INPUT`
- `0` `DECODE_ERROR`

Output nằm ở:

- [manifest.csv](C:/Users/USER/Desktop/ai_detector_img/data/processed_v4_rgb248_r4_exact/manifest.csv)
- [preprocessing_run_summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/preprocessing_run_v4_rgb248_r4_exact/preprocessing_run_summary.json)

### 3.3. Kết quả feature extraction v2

Run full feature extraction trên toàn bộ tập `ACCEPTED` tạo ra:

- `85,615` hàng
- `36` cột đặc trưng
- `0` lỗi extraction
- `0` giá trị `NaN`
- `0` giá trị không hữu hạn

Output nằm ở:

- [feature_extraction_v2_rgb248_exact.csv](C:/Users/USER/Desktop/ai_detector_img/features/feature_extraction_v2_rgb248_exact.csv)
- [feature_extraction_summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/feature_extraction_v2_rgb248_exact/feature_extraction_summary.json)

### 3.4. Kết quả training baseline hiện tại

Training baseline đầu tiên trên full feature table chọn:

- `full_v2__lightgbm`

Kết quả chính:

- `val AUC = 0.9548`
- `id_test AUC = 0.9491`
- `ood_eval AUC = 0.9676`

Tuy nhiên, đây **chưa phải champion-safe** vì model thắng đang dùng mạnh các family:

- `content_adaptive_y_srm`
- `conditional_cfa`
- `wavelet_decay`

Xem:

- [training_baseline_validation.md](training_baseline_validation.md)
- [selected_model_family_importance.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_baseline_20260403/selected_model_family_importance.csv)

## 4. Các vấn đề khoa học cốt lõi mà nhánh này đã làm rõ

### 4.1. Shortcut lớn nhất không nằm ở classifier, mà nằm ở dữ liệu

Từ đầu dự án, ảnh thật và ảnh AI không cùng một phân phối quan sát:

- ảnh thật: JPEG, có lịch sử nén và subsampling thật,
- ảnh AI: PNG, thường không có history tương đương.

Do đó:

- nếu mô hình học được khác biệt giữa `JPEG` và `PNG`, nó có thể đạt AUC cao mà không hề hiểu generator trace,
- nếu tiền xử lý “chuẩn hóa” sai cách, nó còn có thể tạo thêm shortcut mới.

### 4.2. Không thể xóa hoàn toàn history nén bằng một phép biến đổi tất định an toàn

Đây là kết luận xuyên suốt của nhánh này.

Lý do:

- history nén đã in vào pixel domain,
- không có phép tất định class-independent nào vừa xóa hoàn toàn history đó, vừa giữ trọn vẹn mọi bằng chứng pháp y hữu ích,
- các phép như `JPEG bottleneck`, `resize`, `chroma420 canonicalization` thường chỉ đổi hình của nuisance chứ không triệt tiêu nó.

Vì vậy, nhánh này chốt nguyên tắc:

- **preprocessing chỉ được làm những gì trung thực với dữ liệu**,
- phần còn lại phải giải bằng **feature governance** và **audit ở mức model**.

### 4.3. “Không có feature hoàn hảo” không có nghĩa “không còn gì để học”

Một kết quả quan trọng của nhánh là:

- `0` feature đơn vượt qua strict gate,
- `0` feature-set ban đầu vượt qua champion gate cũ.

Điều này không có nghĩa DSP features vô dụng.
Ý nghĩa đúng là:

- cách đánh giá cũ đã đòi hỏi quá nhiều ở một feature đơn,
- nhiều cue pháp y hợp lệ chỉ nên dùng theo chế độ `conditional`,
- quyết định cuối phải ở mức **branch / fusion / system**, không phải feature đơn.

## 5. Những quyết định kiến trúc đã được khóa

### 5.1. Quyết định ở pha tiền xử lý

Champion preprocessing hiện tại là:

- decode canonical
- áp EXIF orientation
- `RGB -> RGB`
- `RGBA -> RGB` bằng `straight-alpha composite` trên nền xám `128`
- loại `L`, `CMYK` và mode chưa audit
- exact crop `248x248 @ residue (4,4)`
- không padding
- không resize
- không JPEG bottleneck
- không chroma canonicalization

Mục tiêu của lựa chọn này là:

- giảm tối đa shortcut do **chính pipeline tạo ra**,
- đồng thời không phá hủy thêm bằng chứng pixel thật.

### 5.2. Quyết định ở pha feature extraction

Feature space hiện hành của nhánh là một hệ `multi-branch`:

- `always-on`
  - `control_minimal`
  - `fft_midband_y`

- `conditional`
  - `conditional_cfa_rgb`

- `research-only`
  - `wavelet_decay`
  - `dark_textured_hetero`
  - `content_adaptive_y_srm`

- `drop`
  - các family codec-direct hoặc chroma-toxic

Điểm khác biệt lớn với baseline cũ:

- không còn cố cứu toàn bộ 33 feature legacy,
- không giả định mọi feature phải luôn luôn bật,
- chấp nhận rằng một số cue vật lý chỉ hợp lệ khi điều kiện tần số còn tồn tại.

### 5.3. Quyết định ở pha training

Pha training hiện chỉ được xem là:

- **baseline benchmark có kiểm soát**,
- chưa phải huấn luyện champion cuối cùng.

Lý do:

- selected model hiện còn dùng mạnh các family chưa đủ nuisance-audit,
- benchmark hiện mới mạnh trên clean split,
- chưa có model-level `AUC_nat` và `AUC_xdeg`.

## 6. Dòng thời gian phiên bản của nhánh

Lịch sử Git của nhánh hiện tại:

| Commit | Vai trò |
|---|---|
| `c683ffc` | snapshot an toàn trước khi rewrite pha feature extraction; chứa trạng thái tiền xử lý v4, artifact và tài liệu nền của nhánh |
| `31da5da` | rewrite toàn bộ `src/feature_extraction` theo spec v2 |
| `b7b4e55` | siết input contract và harden runtime guard cho feature extraction |
| `628dd7d` | thêm package training baseline v2, notebook training mới và benchmark baseline đầu tiên |

Điểm cần lưu ý:

- phần lớn groundwork của preprocessing v4 được chốt **trước** snapshot `c683ffc`,
- từ commit này trở đi, nhánh bước sang giai đoạn “ổn định preprocessing, viết lại feature extraction, rồi benchmark training”.

## 7. Trạng thái hiện tại của nhánh

### 7.1. Những gì đã hoàn thành

- đã khóa preprocessing `v4_exact`
- đã xử lý mode asymmetry (`RGBA`, `L`, `CMYK`) theo hướng fail-closed
- đã rewrite feature extraction `v2`
- đã chạy full extraction trên toàn bộ tập `ACCEPTED`
- đã có training baseline đầu tiên trên full feature table

### 7.2. Những gì chưa thể khẳng định

- chưa thể nói selected model hiện tại học đúng “generator trace” nhiều hơn “nuisance” ở mức hệ thống,
- chưa thể chốt threshold deploy cuối,
- chưa thể chốt champion model.

### 7.3. Việc phải làm tiếp theo

1. audit model-level `AUC_nat`
2. audit model-level `AUC_xdeg`
3. ablation theo family:
   - bỏ `Y-SRM`
   - bỏ `CFA`
   - bỏ `wavelet`
4. đo lại coverage và fairness của `conditional CFA gate`
5. chỉ sau đó mới quyết định có đi tiếp tới champion training hay phải quay lại feature governance

## 8. Thứ tự đọc khuyến nghị cho người mới

Nếu chưa biết gì về dự án, nên đọc theo thứ tự:

1. tài liệu này: [branch_experiment_overview.md](branch_experiment_overview.md)
2. spec tiền xử lý: [preprocessing_standard.md](../../specs/active/preprocessing_standard.md)
3. báo cáo validation tiền xử lý: [preprocessing_validation_summary.md](preprocessing_validation_summary.md)
4. spec feature extraction: [feature_extraction_standard.md](../../specs/active/feature_extraction_standard.md)
5. báo cáo validation feature: [feature_space_validation_summary.md](feature_space_validation_summary.md)
6. spec training/eval: [training_evaluation_standard.md](../../specs/active/training_evaluation_standard.md)
7. báo cáo training baseline: [training_baseline_validation.md](training_baseline_validation.md)

## 9. Kết luận ngắn

Nhánh này đã chuyển bài toán từ trạng thái:

- “mô hình có vẻ tốt nhưng không biết đang học gì”

thành trạng thái:

- “pipeline đã sạch hơn nhiều, feature space đã được cấu trúc hóa, baseline đã chạy được, và giờ có thể audit model ở mức hệ thống”.

Đây là bước tiến lớn nhất của nhánh.
Điểm nghẽn hiện tại không còn là viết code cho preprocessing hay feature extraction,
mà là **chứng minh được model cuối cùng học sự khác biệt thật giữa ảnh diffusion và ảnh real, thay vì tiếp tục học compression history**.
