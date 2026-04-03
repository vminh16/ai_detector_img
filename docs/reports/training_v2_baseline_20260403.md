# Training v2 Baseline (2026-04-03)

## Mục tiêu

Báo cáo này trả lời ba câu hỏi sau trên full feature table
[`feature_extraction_v2_rgb248_exact.csv`](../../features/feature_extraction_v2_rgb248_exact.csv):

- branch/feature-set nào còn utility thực sự ở pha training
- mô hình tuyến tính hay phi tuyến đang tận dụng tốt hơn stack hiện tại
- có thể chuyển sang champion training ngay hay vẫn cần governance experiment bổ sung

Artifact số liệu của vòng này nằm ở:

- [summary.json](../../audit_output/validation/training_v2_baseline_20260403/summary.json)
- [candidate_val_metrics.csv](../../audit_output/validation/training_v2_baseline_20260403/candidate_val_metrics.csv)
- [selected_model_metrics.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_metrics.csv)
- [selected_model_ood_by_generator.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_ood_by_generator.csv)
- [selected_model_feature_importance.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_feature_importance.csv)
- [selected_model_family_importance.csv](../../audit_output/validation/training_v2_baseline_20260403/selected_model_family_importance.csv)
- [cfa_gate_coverage.csv](../../audit_output/validation/training_v2_baseline_20260403/cfa_gate_coverage.csv)

## Dữ liệu đầu vào

- số hàng: `85,615`
- `feature_version = v2_rgb248_exact_multibranch`
- `preprocess_version = v4_rgb248_r4_exact`
- `0` lỗi extraction
- `0` NaN
- `0` non-finite cells

Split sử dụng:

- `train_core = 44,235`
- `calibration = 2,328`
- `val = 5,821`
- `id_test = 5,821`
- `ood_eval = 27,410`

## Thiết kế benchmark

Feature sets:

- `control_minimal`
- `always_on`
- `always_on_plus_cfa_raw`
- `always_on_plus_cfa_gated`
- `full_v2`

Models:

- `logreg`
- `lightgbm`

Protocol:

1. fit base model trên `train_core`
2. fit Platt scaling trên `calibration`
3. chọn model theo `val_auc`
4. khóa threshold trên `val` với ràng buộc `FPR <= 5%`
5. chỉ sau đó mới báo `id_test` và `ood_eval`

## Kết quả candidate trên validation

Top candidates:

| Candidate | Val AUC | Val Brier | Val ECE | Val TPR @ 5% FPR |
|---|---:|---:|---:|---:|
| `full_v2__lightgbm` | `0.9548` | `0.0832` | `0.0145` | `0.7953` |
| `always_on_plus_cfa_raw__lightgbm` | `0.9340` | `0.1025` | `0.0128` | `0.7260` |
| `full_v2__logreg` | `0.9140` | `0.1163` | `0.0182` | `0.6757` |
| `always_on_plus_cfa_raw__logreg` | `0.9046` | `0.1235` | `0.0155` | `0.6527` |

Bottom of the table:

- `always_on__lightgbm`: `AUC = 0.8214`
- `control_minimal__lightgbm`: `AUC = 0.7730`
- `control_minimal__logreg`: `AUC = 0.7085`

## Selected baseline

Candidate được chọn theo `val_auc` là:

- `full_v2__lightgbm`

Operating threshold khóa trên `val`:

- `threshold = 0.7074744498826763`

### Split metrics

| Split | AUC | 95% CI | Brier | ECE | TPR @ locked threshold | FPR @ locked threshold |
|---|---:|---:|---:|---:|---:|---:|
| `val` | `0.9548` | `[0.9508, 0.9595]` | `0.0832` | `0.0145` | `0.7953` | `0.0500` |
| `id_test` | `0.9491` | `[0.9448, 0.9542]` | `0.0897` | `0.0109` | `0.7747` | `0.0518` |
| `ood_eval` | `0.9676` | `[0.9659, 0.9694]` | `0.0704` | `0.0241` | `0.8517` | `0.0502` |

OOD breakdown:

| Generator | AUC | TPR @ locked threshold | FPR @ locked threshold |
|---|---:|---:|---:|
| `GLIDE` | `0.9816` | `0.9153` | `0.0514` |
| `SDv15` | `0.9569` | `0.8039` | `0.0494` |

## Diễn giải đúng của kết quả

Kết quả clean training baseline là mạnh.
Điều này cho thấy feature table `v2` đủ giàu để học trên split clean hiện tại.

Tuy nhiên, kết quả này **không đủ** để tuyên bố champion-safe vì ba lý do:

1. benchmark hiện mới là clean `val/id_test/ood_eval`
   - chưa có `model-level AUC_nat`
   - chưa có `model-level AUC_xdeg`
   - chưa có stress test trên degradation suite sau khi model đã fit

2. model thắng là `full_v2`, không phải `always_on`
   - nghĩa là performance hiện tại phụ thuộc mạnh vào các family còn ở trạng thái `research-only`
   - do đó clean AUC cao không đồng nghĩa stack đã sạch shortcut

3. branch `conditional CFA` hiện còn lệch coverage
   - trên `ood_eval`, `cfa_gate_active` chỉ là `8.72%` với `ai`
   - nhưng là `26.87%` với `nature`
   - đây là dấu hiệu class-conditional coverage shift, chưa an toàn để fuse trực tiếp

## Feature importance của selected model

Top features:

1. `cfa_validity_score`
2. `kurt_noise_y`
3. `spatial_snr_ratio`
4. `energy_ratio_chroma`
5. `ysrm_midtex_square5_energy`
6. `ysrm_midtex_square3_mar`
7. `wav_ratio_l1_l2`

Family importance cộng gộp:

| Family | Importance |
|---|---:|
| `content_adaptive_y_srm` | `2531` |
| `conditional_cfa` | `1915` |
| `fft_midband` | `1763` |
| `control_spatial` | `1540` |
| `wavelet_decay` | `1338` |
| `control_color` | `1280` |
| `dark_textured_hetero` | `1270` |
| `control_frequency` | `363` |

Diễn giải:

- selected model đang dùng mạnh `Y-SRM`, `CFA`, `wavelet`
- đây chính là các family hiện chưa được khóa là champion-safe trong spec v2
- vì vậy kết quả clean tốt hiện tại nhiều khả năng đang đến từ các cue mạnh nhưng chưa được nuisance-audit đủ sâu

## Phán quyết

Có thể chuyển sang pha training baseline.
Chưa nên chuyển sang champion training cuối.

Lý do:

- code extraction và full feature table đã đủ sạch để train
- baseline model cho thấy stack hiện tại có signal mạnh
- nhưng selected model hiện vẫn phụ thuộc đáng kể vào các family `research-only / conditional`
- do đó cần thêm governance experiment ở **mức model**, không chỉ mức feature

## Việc phải làm tiếp theo

1. chạy `model-level AUC_nat`
   - đặc biệt cho `full_v2__lightgbm`
   - và so sánh với `always_on__lightgbm`

2. chạy `model-level AUC_xdeg`
   - `jpeg95_420`
   - `jpeg90_420`
   - `resize75_bilinear`
   - `resize50_bilinear`
   - `resize50_jpeg90_420`

3. chạy branch ablation có kiểm soát
   - `full_v2`
   - `full_v2 - YSRM`
   - `full_v2 - CFA`
   - `full_v2 - wavelet`
   - `always_on + one research family`

4. nếu `full_v2` chỉ thắng nhờ `Y-SRM/CFA` nhưng sụp mạnh trên `nat/xdeg`, phải quay lại governance
5. nếu `full_v2` vẫn giữ được margin trên `nat/xdeg`, lúc đó mới đáng cân nhắc champion training và threshold locking cuối
