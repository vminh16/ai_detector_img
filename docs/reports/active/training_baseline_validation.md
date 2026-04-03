# Báo cáo training baseline v2

## 1. Mục tiêu

Báo cáo này là report active cho pha training baseline trên full feature table của nhánh.

Nó trả lời bốn câu hỏi:

1. liệu feature table `v2` đã đủ để train baseline hay chưa,
2. branch/feature-set nào mạnh nhất trên clean split,
3. mô hình tuyến tính hay phi tuyến đang tận dụng feature space tốt hơn,
4. có thể sang champion training ngay hay vẫn phải audit tiếp ở mức model.

Tài liệu này được khóa bởi:

- [candidate_val_metrics.csv](../../audit_output/validation/training_v2_baseline_20260403/candidate_val_metrics.csv)
- [selected_model_metrics.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_metrics.csv)
- [selected_model_ood_by_generator.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_ood_by_generator.csv)
- [selected_model_feature_importance.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_feature_importance.csv)
- [selected_model_family_importance.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_family_importance.csv)
- [cfa_gate_coverage.csv](../../audit_output/validation/training_v2_baseline_20260403/cfa_gate_coverage.csv)

## 2. Dữ liệu đầu vào và protocol

Input là:

- [feature_extraction_v2_rgb248_exact.csv](../../features/feature_extraction_v2_rgb248_exact.csv)

với:

- `85,615` hàng
- `36` cột đặc trưng
- `0` lỗi
- `0` `NaN`

Split sử dụng:

- `train_core`
- `calibration`
- `val`
- `id_test`
- `ood_eval`

Protocol huấn luyện:

1. fit model trên `train_core`
2. fit Platt scaling trên `calibration`
3. chọn candidate theo `val_auc`
4. khóa threshold trên `val` với ràng buộc `FPR <= 5%`
5. chỉ sau đó mới báo `id_test` và `ood_eval`

Điểm quan trọng:

- `id_test` và `ood_eval` không được dùng để chọn candidate,
- vì vậy kết quả ở hai split này mới có ý nghĩa kiểm chứng.

## 3. Candidate benchmark

Các feature-set đã benchmark:

- `control_minimal`
- `always_on`
- `always_on_plus_cfa_raw`
- `always_on_plus_cfa_gated`
- `full_v2`

Các model:

- `logreg`
- `lightgbm`

Kết quả validation cho thấy:

- mọi candidate dùng `lightgbm` đều mạnh hơn `logreg`,
- `full_v2__lightgbm` đứng đầu bảng,
- `control_minimal` vẫn quá yếu để làm branch duy nhất.

Điều này phù hợp với kết luận của pha feature:

- multi-feature fusion có ích thật,
- nonlinear model tận dụng tương tác feature tốt hơn,
- nhưng performance cao chưa nói được model đang học signal sạch hay signal bẩn.

## 4. Kết quả của baseline tốt nhất

Model thắng hiện tại:

- `full_v2__lightgbm`

Kết quả:

| Split | AUC | Brier | ECE | TPR tại threshold khóa | FPR tại threshold khóa |
|---|---:|---:|---:|---:|---:|
| `val` | `0.9548` | `0.0832` | `0.0145` | `0.7953` | `0.0500` |
| `id_test` | `0.9491` | `0.0897` | `0.0109` | `0.7747` | `0.0518` |
| `ood_eval` | `0.9676` | `0.0704` | `0.0241` | `0.8517` | `0.0502` |

OOD theo generator:

| Generator | AUC | TPR | FPR |
|---|---:|---:|---:|
| `GLIDE` | `0.9816` | `0.9153` | `0.0514` |
| `SDv15` | `0.9569` | `0.8039` | `0.0494` |

Kết luận ngắn:

- feature table hiện tại đủ mạnh để train baseline tốt,
- OOD clean trên hai generator hold-out không sụp,
- calibration hiện cũng khá gọn.

## 5. Vì sao kết quả này chưa đủ để gọi là champion-safe

Đây là điểm quan trọng nhất của report.

### 5.1. Benchmark hiện mới là clean benchmark

Hiện tại chưa có:

- `model-level AUC_nat`
- `model-level AUC_xdeg`

Nghĩa là:

- model thắng có thể rất mạnh trên clean split,
- nhưng vẫn có thể hút mạnh nuisance ở mức hệ thống.

### 5.2. Selected model dùng mạnh các family chưa đủ sạch

Feature importance hiện tại cho thấy:

- `cfa_validity_score` là feature quan trọng nhất
- nhiều feature `Y-SRM` đứng rất cao
- `wavelet` cũng được dùng đáng kể

Family importance cộng gộp:

1. `content_adaptive_y_srm`
2. `conditional_cfa`
3. `fft_midband`
4. `control_spatial`
5. `wavelet_decay`

Điều này có nghĩa:

- selected model không chỉ sống nhờ `always-on`,
- nó đang tận dụng mạnh các family còn ở trạng thái `research-only / conditional`.

Do đó:

- clean AUC cao hiện tại không thể dùng như bằng chứng rằng pipeline đã “học đúng bản chất”.

### 5.3. CFA gate hiện còn lệch coverage theo lớp

Audit gate coverage cho thấy:

- ở `ood_eval`, `cfa_gate_active` chỉ khoảng `8.7%` với `ai`
- nhưng khoảng `26.9%` với `nature`

Đây là dấu hiệu:

- validity gate chưa thật sự class-independent ở mức hệ thống,
- hoặc ít nhất đang tương tác mạnh với phân phối OOD hiện tại.

Nếu không audit tiếp, fusion có thể học chính pattern coverage này.

## 6. Ý nghĩa thực sự của baseline hiện tại

Baseline này rất có giá trị, nhưng giá trị đúng của nó là:

- chứng minh full feature table `v2` có signal mạnh,
- chứng minh nonlinear fusion đáng để tiếp tục,
- cung cấp một mô hình thực để làm `nuisance stress test`.

Giá trị của nó **không phải** là:

- chứng minh selected model đủ sạch để release,
- hay chứng minh spec feature hiện tại đã hoàn chỉnh.

## 7. Phán quyết

Có thể chuyển sang pha training baseline và branch benchmark ngay.
Điều đó đã hoàn thành.

Chưa nên:

- khóa champion model,
- khóa threshold deploy cuối,
- hay kết luận branch hiện tại đã giải xong bài toán shortcut.

## 8. Việc bắt buộc phải làm tiếp theo

1. chạy `model-level AUC_nat`
   - ít nhất cho:
     - `full_v2__lightgbm`
     - `always_on__lightgbm`
     - `always_on_plus_cfa_raw__lightgbm`

2. chạy `model-level AUC_xdeg`
   - `jpeg95_420`
   - `jpeg90_420`
   - `resize75_bilinear`
   - `resize50_bilinear`
   - `resize50_jpeg90_420`

3. chạy ablation theo family
   - `full_v2 - YSRM`
   - `full_v2 - CFA`
   - `full_v2 - wavelet`
   - `full_v2 - dark_textured_hetero`

4. nếu selected model sụp mạnh trên `AUC_nat` hoặc `AUC_xdeg`
   - quay lại governance
   - không chuyển sang champion training

5. chỉ nếu selected model giữ được lợi thế sau các stress test này
   - mới hợp lý để làm vòng champion training tiếp theo

## 9. Kết luận ngắn

Training baseline hiện tại là một **bài test hệ thống thành công**,
không phải một **phán quyết cuối cùng**.

Nó cho thấy:

- code đã đủ sạch để train,
- feature table đã đủ giàu để học,
- nhưng câu hỏi quan trọng nhất vẫn còn ở phía trước:

> model hiện tại đang học sự khác biệt thật giữa ảnh diffusion và ảnh real,
> hay chỉ đang tận dụng một hỗn hợp mạnh của signal thật và nuisance còn sót?
