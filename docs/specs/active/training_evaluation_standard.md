# Đặc tả huấn luyện và đánh giá hiện hành

## 1. Trạng thái và phạm vi

Đây là đặc tả active cho pha training/evaluation của nhánh `codex/preprocessing-v4-core`.

Phiên bản triển khai hiện tại:

- **mã nguồn**: [src/training](C:/Users/USER/Desktop/ai_detector_img/src/training)
- **notebook orchestration**: [03_training_eval.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/03_training_eval.ipynb)
- **baseline benchmark active**: `training_v2_baseline_20260403`

Tài liệu này hợp nhất:

- [training_v2_baseline_20260403.md](../../reports/archive/training_v2_baseline_20260403.md)
- [training_baseline_validation.md](../../reports/active/training_baseline_validation.md)

## 2. Mục tiêu của pha này

Pha training/evaluation hiện tại có hai mục tiêu:

1. benchmark có kiểm soát trên feature table `v2`
2. quyết định xem branch nào đáng để đưa sang champion candidate

Pha này **chưa** có mục tiêu:

- chốt champion model cuối,
- chốt threshold deploy cuối,
- hay tuyên bố bài toán shortcut đã được giải xong.

## 3. Input contract

Input duy nhất của pha này là feature table:

- [feature_extraction_v2_rgb248_exact.csv](../../features/feature_extraction_v2_rgb248_exact.csv)

Yêu cầu:

- `feature_version = v2_rgb248_exact_multibranch`
- `preprocess_version = v4_rgb248_r4_exact`
- `status = ok` cho toàn bộ hàng dùng huấn luyện

Nếu không thỏa ba điều kiện trên:

- pipeline phải fail-closed ngay.

## 4. Split contract

Split active:

- `train_core`
- `calibration`
- `val`
- `id_test`
- `ood_eval`

Ý nghĩa:

- `train_core`: fit mô hình gốc
- `calibration`: fit Platt scaling
- `val`: chọn candidate và khóa threshold
- `id_test`: kiểm tra mù trên generator in-domain holdout
- `ood_eval`: kiểm tra mù trên generator out-of-domain holdout

Quy tắc cứng:

- không dùng `id_test` hay `ood_eval` để chọn candidate,
- không fit calibration trên `val`,
- không chỉnh threshold bằng `id_test` hoặc `ood_eval`.

## 5. Candidate benchmark

Feature-set benchmark active:

- `control_minimal`
- `always_on`
- `always_on_plus_cfa_raw`
- `always_on_plus_cfa_gated`
- `full_v2`

Model family active:

- `logreg`
- `lightgbm`

Điểm quan trọng:

- phase này cho phép so sánh model tuyến tính và phi tuyến,
- nhưng không được dùng model phi tuyến để hợp thức hóa direct proxy.

## 6. Calibration và threshold lock

### 6.1. Calibration

Calibration active dùng:

- base model fit trên `train_core`
- `Platt scaling` fit trên `calibration`

Output score chính thức của baseline là:

- xác suất sau calibration

### 6.2. Threshold lock

Threshold được khóa trên `val` theo nguyên tắc:

- tìm threshold nhỏ nhất sao cho `FPR_val <= 5%`

Threshold đã khóa chỉ được:

- áp sang `id_test`
- áp sang `ood_eval`

Không được lock threshold riêng cho từng split sau.

## 7. Metric bắt buộc

Mỗi candidate và selected model phải báo:

- `AUC`
- `Brier`
- `ECE`
- `TPR`
- `FPR`
- `precision`
- `accuracy`
- `threshold`

Với selected model, phải báo thêm:

- bootstrap `95% CI` cho `AUC`
- breakdown theo generator ở `ood_eval`
- feature importance
- family importance

## 8. Metric còn thiếu nhưng bắt buộc ở vòng tiếp theo

Training baseline hiện tại **chưa đủ** nếu thiếu:

- `model-level AUC_nat`
- `model-level AUC_xdeg`
- branch/family ablation

Đây là requirement bắt buộc trước khi gọi bất kỳ model nào là champion candidate.

## 9. Kết quả baseline hiện tại

Selected baseline hiện tại là:

- `full_v2__lightgbm`

Kết quả:

- `val AUC = 0.9548`
- `id_test AUC = 0.9491`
- `ood_eval AUC = 0.9676`

Kết luận đúng của kết quả này:

- feature table `v2` đủ mạnh để train baseline,
- nonlinear fusion đáng để tiếp tục,
- nhưng selected model vẫn chưa champion-safe.

## 10. Vì sao chưa được coi là champion-safe

Có ba lý do:

1. chưa có `model-level AUC_nat` và `AUC_xdeg`
2. selected model dùng mạnh các family còn ở `research-only / conditional`
3. `conditional CFA gate` hiện còn lệch coverage theo lớp trên `ood_eval`

Do đó:

- clean benchmark tốt là tín hiệu tích cực,
- nhưng chưa phải phán quyết cuối.

## 11. Champion-readiness criteria

Một model chỉ được nâng từ baseline lên champion candidate khi đồng thời:

1. giữ lợi thế trên `val`, `id_test`, `ood_eval`
2. không sụp trên `AUC_nat`
3. không sụp trên `AUC_xdeg`
4. không phụ thuộc chủ yếu vào branch có coverage lệch hoặc nuisance chưa audit
5. có calibration ổn định trên các split chính

Nếu không thỏa các điều kiện trên:

- model chỉ được giữ ở trạng thái benchmark/reference,
- không được dùng để khóa threshold deploy.

## 12. Thứ tự thí nghiệm kế tiếp

Pha tiếp theo phải chạy theo thứ tự:

1. `model-level AUC_nat`
2. `model-level AUC_xdeg`
3. family ablation
4. so sánh lại:
   - `always_on`
   - `always_on + CFA`
   - `always_on + wavelet`
   - `always_on + Y-SRM`
   - `full_v2 - từng family`
5. chỉ sau đó mới xem xét champion training

## 13. Kết luận

Training/evaluation hiện hành là **pha benchmark có kiểm soát ở mức hệ thống**.

Nó không còn trả lời câu hỏi:

- “model nào AUC cao nhất?”

Mà trả lời câu hỏi khó hơn:

- “model nào còn giữ được utility khi phải đối mặt với nuisance mà nhánh này đã phát hiện?”
