# AI Detector Image

Repo nghiên cứu và tái cấu trúc pipeline phát hiện ảnh AI trên dữ liệu web in-the-wild, với trọng tâm là:

- loại bỏ shortcut do `JPEG / PNG / chroma subsampling / image mode`
- chuẩn hóa lại pha tiền xử lý và trích chọn đặc trưng
- kiểm tra mô hình ở mức hệ thống, không chỉ nhìn `AUC` clean

Trạng thái hiện tại của repo:

- đây là **nhánh thực nghiệm đã tái cấu trúc mạnh**
- pipeline đã chạy đủ ba pha: `preprocessing`, `feature extraction`, `training/evaluation`
- đã có benchmark reference mạnh trên clean
- **chưa có champion model đủ an toàn để coi là production-ready**

## 1. Bài toán của repo

Bài toán cốt lõi không chỉ là phân loại `real` và `AI`.

Với dữ liệu ảnh web hiện tại:

- ảnh thật chủ yếu là `JPEG`
- ảnh AI chủ yếu là `PNG`
- nhiều đặc trưng pháp y cổ điển phản ứng trực tiếp với `compression history`

Nếu pipeline làm sai, mô hình có thể đạt điểm rất cao nhưng thực chất chỉ học:

- định dạng ảnh
- lịch sử nén
- chroma subsampling
- hoặc mode ảnh như `RGBA`, `L`

Nhánh thực nghiệm này được tạo ra để giải đúng vấn đề đó.

## 2. Kết quả chính của nhánh hiện tại

### Tiền xử lý

Pipeline active hiện tại là `v4_exact`:

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

### Trích chọn đặc trưng

Feature extraction active hiện tại là `v2_rgb248_exact_multibranch`:

- `always-on`
- `conditional CFA`
- `research branches` cho các cue còn rủi ro
- loại bỏ các family codec-direct đã bị chứng minh là độc

### Huấn luyện và đánh giá

Notebook train đã chạy đủ:

1. clean benchmark
2. model-level nuisance audit (`AUC_nat`)
3. degradation suite (`AUC_xdeg`)
4. family ablation
5. phase closure

Model benchmark mạnh nhất hiện tại là `full_v2__lightgbm`, nhưng chưa được coi là champion vì:

- `AUC_nat_abs` vẫn còn cao
- sụp mạnh dưới `resize-heavy degradation`
- còn phụ thuộc nhiều vào các branch điều kiện / research

## 3. Cấu trúc repo

```text
app/          demo/API đơn giản
deploy/       pipeline chạy suy luận / phục vụ demo
docs/         đặc tả, báo cáo, tài liệu tham khảo
inference/    runtime inference cũ và công cụ liên quan
notebooks/    notebook theo từng pha
script/       script audit, validation, studies, notebook builders
src/          mã nguồn lõi đã tái cấu trúc
```

Các thư mục quan trọng trong `src/`:

- `src/preprocessing/`: pipeline tiền xử lý canonical
- `src/feature_extraction/`: pipeline trích chọn đặc trưng nhiều nhánh
- `src/training/`: benchmark, calibration, nuisance audit, degradation suite
- `src/visualization/`: hàm trực quan hóa theo từng pha
- `src/dataset_tools/`: tiện ích xử lý dữ liệu

## 4. Tài liệu nên đọc trước

Nếu muốn hiểu nhanh toàn bộ repo, nên đọc theo thứ tự:

1. [docs/reports/active/branch_experiment_overview.md](docs/reports/active/branch_experiment_overview.md)
2. [docs/specs/active/preprocessing_standard.md](docs/specs/active/preprocessing_standard.md)
3. [docs/specs/active/feature_extraction_standard.md](docs/specs/active/feature_extraction_standard.md)
4. [docs/specs/active/training_evaluation_standard.md](docs/specs/active/training_evaluation_standard.md)
5. [docs/reports/active/preprocessing_validation_summary.md](docs/reports/active/preprocessing_validation_summary.md)
6. [docs/reports/active/feature_space_validation_summary.md](docs/reports/active/feature_space_validation_summary.md)
7. [docs/reports/active/training_baseline_validation.md](docs/reports/active/training_baseline_validation.md)

`docs/README.md` là bản đồ tài liệu tổng quát của toàn repo.

## 5. Quy trình chạy theo thứ tự

### Pha 1: Tiền xử lý

Notebook:

- [notebooks/01_preprocessing.ipynb](notebooks/01_preprocessing.ipynb)

Source-of-truth:

- [src/preprocessing](src/preprocessing)

Output mong đợi:

- manifest canonical patch
- patch `RGB 248x248`
- audit preprocessing

### Pha 2: Trích chọn đặc trưng

Notebook:

- [notebooks/02_feature_extraction.ipynb](notebooks/02_feature_extraction.ipynb)

Source-of-truth:

- [src/feature_extraction](src/feature_extraction)

Output mong đợi:

- feature table `v2`
- validation extraction

### Pha 3: Huấn luyện và đánh giá

Notebook:

- [notebooks/03_training_eval.ipynb](notebooks/03_training_eval.ipynb)

Source-of-truth:

- [src/training](src/training)

Output mong đợi:

- clean benchmark
- `AUC_nat`
- degradation suite
- family ablation
- phase closure summary

## 6. Cài đặt môi trường

Repo hiện dùng Python và notebook. Cách đơn giản nhất:

```bash
pip install -r requirements.txt
```

Nếu chạy notebook, nên dùng một môi trường ảo riêng để tránh lệch phiên bản thư viện.

## 7. Dữ liệu và artifact

Repo này có nhiều output sinh ra từ notebook và validation:

- `data/`
- `features/`
- `models/`
- `audit_output/`

Các thư mục này có thể rất lớn. Trong trạng thái local hiện tại:

- `data/` khoảng `46 GB`
- `audit_output/` khoảng `0.24 GB`
- `features/` khoảng `0.14 GB`
- `models/` khoảng `0.10 GB`

Vì vậy:

- code và tài liệu là phần nên đưa lên GitHub
- dữ liệu raw, feature tables, model binaries, và phần lớn artifact benchmark nên để local hoặc tách sang storage khác

## 8. Trạng thái kỹ thuật hiện tại

Repo hiện đã đạt:

- preprocessing đủ chặt để không tự tạo shortcut lớn
- feature extraction đủ ổn định để chạy full corpus
- training notebook đủ để đóng pha benchmark ở mức hệ thống

Repo hiện chưa đạt:

- champion-ready model
- deploy-ready threshold
- bằng chứng cho thấy model đã thoát khỏi `compression history`

Điểm nghẽn hiện tại là:

- `AUC_nat` vẫn cao
- resize làm model sụp mạnh
- các branch utility cao nhất vẫn là các branch nhạy rủi ro

## 9. Repo này phù hợp để làm gì

Phù hợp:

- nghiên cứu forensic pipeline cho ảnh AI
- đọc lại toàn bộ reasoning từ preprocessing tới training
- tái lập benchmark và các study trong repo
- tiếp tục phát triển branch feature/model tiếp theo

Không phù hợp:

- dùng ngay như một detector production
- suy diễn rằng clean `AUC` cao là đủ an toàn

## 10. Ghi chú khi đưa lên GitHub cá nhân

Khuyến nghị:

- đẩy phần `code + docs + notebooks + scripts`
- không đẩy `data/`, model binaries, feature tables và toàn bộ artifact sinh ra
- mô tả rõ trong README rằng đây là repo nghiên cứu, không phải bản release production

Nếu muốn public repo:

- nên giữ `docs/specs/active/` và `docs/reports/active/` làm source-of-truth
- phần `archive/` giữ lại để người đọc truy vết lịch sử quyết định

## 11. Giấy phép

Repo hiện chưa khai báo `LICENSE`. Nếu định public lâu dài, nên thêm giấy phép phù hợp trước khi công khai.
