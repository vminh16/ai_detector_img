# Feature Extraction Standard v0

> Superseded by [feature_extraction_standard_v1.md](C:/Users/USER/Desktop/ai_detector_img/docs/specs/feature_extraction_standard_v1.md).

## 1. Status và mục tiêu

Tài liệu này là bản nháp active cho pha trích xuất đặc trưng sau
`preprocessing_pipeline_standard_v4.md`.

Bản `v0` đã được sửa lại sau study đối sánh trực tiếp giữa:

- `old_v1`: preprocessing cũ `256 x 256`, có pad / conditional shift / JPEG bottleneck
- `v4_exact`: preprocessing mới `248 x 248 @ residue (4,4)`, exact crop, không pad, không resize, không bottleneck

Mục tiêu của `feature_extract_v0` là:

- chỉ nhận đầu vào canonical `X_can_rgb8`, shape `(248, 248, 3)`, `uint8`
- giảm tối đa khả năng học `JPEG history`, `chroma subsampling history`, và `post-process history`
- giữ lại các image-formation traces có ý nghĩa vật lý và còn sống sót sau web pipeline
- tách rõ `champion-safe core` khỏi `priority-audit` và `research-only`

## 2. Nguồn bằng chứng bắt buộc

Spec này được khóa bởi các tài liệu và artifact sau:

- `docs/specs/preprocessing_pipeline_standard_v4.md`
- `docs/reports/shortcut_risk_validation.md`
- `docs/reports/feature_space_update.md`
- `docs/reports/feature_spec_v0_review.md`
- `docs/reports/feature_spec_v1_validation.md`
- `audit_output/validation/spec_v4_20260319/feature_governance_summary.json`
- `audit_output/validation/spec_v4_20260319/keep_only_ablation.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/feature_set_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/single_feature_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/feature_shift_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/summary.json`

## 3. Input contract

Feature extractor v0 chỉ được nhận:

- `X_can_rgb8`, shape `(248, 248, 3)`, dtype `uint8`
- patch đã qua orientation, mode policy, support gate, exact crop trong preprocessing v4

Không được dùng:

- file extension
- format gốc
- mode gốc
- alpha flag
- grayscale flag
- metadata / EXIF / ICC
- missingness indicator
- reject-path indicator

Derived views được phép:

- `RGB -> YCrCb` bằng transform tất định duy nhất được khóa trong phase feature
- residual maps
- FFT / radial statistics
- wavelet maps
- CFA residual maps

Derived view không được branch theo source format.

## 4. Nguyên lý toán học và thống kê

### 4.1. Ảnh quan sát đã chứa latent history

Đặt:

- `Z`: latent image-formation signal cần giữ
- `H_c`: latent compression / chroma / post-process history
- `X = T_{H_c}(Z)`: ảnh quan sát trong pixel domain

Khi áp thêm một transform `g`, ta chỉ có:

`g(X) = g(T_{H_c}(Z))`

Do đó, kỳ vọng theo `g` chỉ marginalize được ngẫu nhiên của `g`,
không xóa được `H_c` đã có sẵn trong `X`.

Hệ quả:

- preprocessing không thể "xóa sạch" unknown native JPEG history
- feature phase phải được đặt bài toán `admissibility under nuisance`

### 4.2. DPI là anti-overclaim guard, không phải veto

Với mọi operator lossy class-independent `G`:

`I(T_gen ; G(X)) <= I(T_gen ; X)`

Điều này **không** có nghĩa là mọi projection lossy đều có hại cho classifier.
Nó chỉ có nghĩa:

- operator lossy không được admit bởi intuition
- muốn admit thì phải chứng minh bằng audit rằng nó giảm `I(H_c ; G(X))`
  nhanh hơn mức nó làm giảm `I(T_gen ; G(X))`

Trong spec này, mọi suppressor / projection mới đều phải được đánh giá bằng:

- clean utility
- natural nuisance
- cross-degradation generalization
- feature-shift under degradation

### 4.3. Feature admissibility là bài toán utility-vs-nuisance

Nếu một feature có dạng xấp xỉ:

`phi(X) = phi_gen(X) + phi_hist(H_c) + eps`

thì ERM sẽ học `phi_hist` bất cứ khi `I(Y ; phi_hist(H_c))` lớn.

Trong repo hiện tại, đây là hiện tượng đã được đo trực tiếp:

- legacy `baseline33` trên `old_v1`: `label_logo_clean = 0.7195`
- nhưng `real_jpeg_444_vs_420 = 0.9069`

Nên feature admission phải dựa trên benchmark, không dựa trên giải thích lý thuyết đơn thuần.

## 5. Metric bắt buộc

Mọi feature hoặc feature-set muốn được xét vào champion branch phải báo cáo:

### 5.1. Clean utility

`AUC_logo_clean(Phi)`

- train / test theo `Leave-One-Generator-Out`
- chỉ trên tập accepted chung giữa pipeline cần so sánh

### 5.2. Natural nuisance

`AUC_nat(Phi) = AUC(real 4:4:4 vs real 4:2:0 | Phi)`

Metric này đo trực tiếp khả năng đọc native chroma/JPEG history trong lớp `nature`.

### 5.3. Cross-degradation generalization

Với mọi degradation `g` trong suite:

`AUC_xdeg(Phi, g) = AUC(f_train_clean(Phi(X)), Y on Phi(g(X_test_clean))))`

Trong study hiện tại, suite bắt buộc gồm:

- `jpeg95_420`
- `jpeg90_420`
- `resize75_bilinear`
- `resize50_bilinear`
- `resize50_jpeg90_420`

### 5.4. Feature shift under degradation

Với từng feature `j`:

`Shift_j(g, y) = E[ | z_j(g(X)) - z_j(X) | | Y = y ]`

Trong artifact hiện tại, metric được báo cáo dưới tên:

- `mean_abs_z_shift`

### 5.5. Diễn giải metric

Trong phase thiết kế hiện tại:

- `AUC_logo_clean` cao là tốt
- `AUC_nat` gần `0.5` là tốt
- `AUC_xdeg` gần `AUC_logo_clean` là tốt
- `mean_abs_z_shift` nhỏ là tốt

Không được dùng một metric duy nhất để admit feature.

## 6. Kết quả đối sánh `old_v1` vs `v4_exact`

Study `feature_spec_v1_validation_20260325` được chạy trên:

- `85615` ảnh chung được accept bởi cả `old_v1` và `v4_exact`
- clean LOGO sample: `1400` ảnh cân bằng theo `generator x label`
- real nuisance sample: `1288` ảnh (`700` ảnh `4:4:4`, `588` ảnh `4:2:0`)

### 6.1. `v4_exact` không "làm feature xấu đi"; nó làm lộ rõ utility thật và nuisance thật

Một số set quan trọng:

| Feature set | Preprocess | Clean LOGO AUC | Real 444 vs 420 | JPEG90 xdeg | Resize50 xdeg |
|---|---|---:|---:|---:|---:|
| `baseline33` | `old_v1` | `0.7195` | `0.9069` | - | - |
| `safe_core` | `old_v1` | `0.6890` | `0.6641` | `0.6916` | `0.6227` |
| `safe_core` | `v4_exact` | `0.6861` | `0.6717` | `0.6850` | `0.6016` |
| `wavelet_parent_only` | `old_v1` | `0.6863` | `0.6620` | `0.6874` | `0.6247` |
| `wavelet_parent_only` | `v4_exact` | `0.6891` | `0.6646` | `0.6906` | `0.6247` |
| `cfa_chroma_only` | `old_v1` | `0.5959` | `0.7917` | `0.5647` | `0.5918` |
| `cfa_chroma_only` | `v4_exact` | `0.8158` | `0.8868` | `0.5008` | `0.5834` |
| `crsrm_only` | `old_v1` | `0.6853` | `0.8824` | `0.5790` | `0.6344` |
| `crsrm_only` | `v4_exact` | `0.9236` | `0.9447` | `0.4983` | `0.6039` |

Kết luận:

- `old_v1` đã che mất một phần signal thật của các family nhạy với lattice/history
- `v4_exact` bảo toàn signal đó, đồng thời cũng làm lộ rằng nhiều family đang shortcut-dominated
- spec feature phải được khóa trên `v4_exact`, không khóa trên `old_v1`

### 6.2. Sau khi bỏ các feature độc hại nhất, signal còn lại là có thật nhưng không lớn

`safe_core` trên `v4_exact`:

- `clean = 0.6861`
- `real_jpeg_444_vs_420 = 0.6717`
- `jpeg90_xdeg = 0.6850`
- `resize50_xdeg = 0.6016`

Điều này xác nhận lo ngại sau khi bỏ `DROP`:

- signal an toàn còn lại không lớn
- cần mở rộng feature space
- nhưng không được mở rộng mù

### 6.3. Resize là phép phá hủy nghiêm trọng hơn extra JPEG cho nhiều feature pháp y

Ví dụ trên `v4_exact`, `mean_abs_z_shift` của một số feature:

| Feature | JPEG90 (nature) | Resize50 (nature) |
|---|---:|---:|
| `cfa_cr_pi_xy` | `0.8451` | `1.4756` |
| `cfa_rg_pi_xy` | `0.7974` | `0.8750` |
| `wav_parent_corr_h` | `0.0428` | `1.8546` |
| `wav_parent_corr_v` | `0.0361` | `1.8848` |
| `ps_alpha` | `0.0113` | `2.4193` |

Hệ quả:

- `Cross-Degradation AUC` là metric bắt buộc
- `CFA` không thể được xem là "safe by default"
- patch crop exact chỉ giữ dữ liệu trong lúc trích xuất, không cứu được trace đã bị resize trước đó

## 7. Governance mới cho các family

## 7.1. Champion-safe core

Đây là seed set sạch nhất hiện tại để làm baseline champion branch:

### `safe_core`

Bao gồm:

- `frs_mid_variance`
- `wav_parent_corr_h`
- `wav_parent_corr_v`
- `pearson_y_cr`
- `pearson_y_cb`
- `pearson_cr_cb`
- `energy_ratio_chroma`
- `spatial_snr_ratio`
- `skew_noise_y`
- `kurt_noise_y`

Lý do:

- utility có thật trên `v4_exact`
- nuisance thấp hơn rõ rệt so với chroma texture / CFA directionals
- cross-degradation ổn định nhất trong các set đã đo

Đây **không** phải stack cuối.
Nó là baseline "ít bẩn nhất" để so sánh với các nhánh mạnh hơn nhưng rủi ro hơn.

## 7.2. Priority-audit

Những family này có utility thật, nhưng chưa đủ sạch để admit vào champion:

### `CFA pi_xy subset`

Bao gồm:

- `cfa_cr_pi_xy`
- `cfa_cb_pi_xy`
- `cfa_rg_pi_xy`
- `cfa_bg_pi_xy`

Bằng chứng:

- `cfa_cr_pi_xy` trên `v4_exact`: clean `0.7553`, natural nuisance `0.5114`
- `cfa_rg_pi_xy`: clean `0.7276`, natural nuisance `0.4851`
- nhưng cả hai đều sập gần chance trên `jpeg90_xdeg`

Phán quyết:

- giữ để nghiên cứu tiếp
- không auto-keep
- phải qua thêm `resize stress` và redesign trong RGB domain / edge-gated / flat-gated

### `Y-SRM`

Bao gồm:

- `ysrm_square3_*`
- `ysrm_edge3_*`
- `ysrm_square5_*`

Bằng chứng:

- `ysrm_only` trên `v4_exact`: clean `0.6995`, natural nuisance `0.8348`
- utility có thật, nhưng nuisance vẫn cao

Phán quyết:

- không được drop blanket như chroma SRM
- đưa vào `priority-audit`

### `ps_alpha`

Bằng chứng:

- `clean = 0.7021`
- `natural nuisance = 0.6914`
- `jpeg90_xdeg = 0.7017`
- `resize50_xdeg = 0.6683`

Phán quyết:

- utility có thật
- đọc một phần native history thật
- để `priority-audit / quarantine`, không vào champion-safe core

## 7.3. Research-only

Những hướng này cần thêm implementation hoặc benchmark mới trước khi được xét vào champion:

### `Y-LBP`

Lý do:

- `ylbp_only` trên `v4_exact`: clean `0.7251`, nhưng nuisance `0.8609`
- nhạy với local texture / aliasing quá mạnh để admit ngay

### `NLF` mở rộng

Bằng chứng hiện tại chỉ cho thấy:

- current 5-feature NLF rất yếu (`nlf_only` clean `0.4957-0.5628` tùy pipeline)

Điều này **không** đủ để bác bỏ toàn bộ giả thuyết NLF.
Cần một cụm NLF rộng hơn:

- heteroskedasticity trên flat regions
- luma/chroma noise coupling
- stationarity của noise map
- blockwise slope dispersion

### `Wavelet full`

`wav_parent_corr_h/v` là hữu ích.
Nhưng:

- `wav_parent_corr_d`
- `wav_kurtosis_l1`
- `wav_kurtosis_l2`

chưa cho thấy gain rõ nét.
Giữ để nghiên cứu, không vào champion-safe core.

### Feature mới từ văn học

Cần nghiên cứu thêm:

- chromatic aberration
- PRNU / sensor pattern noise
- resampling periodicity
- robust CFA redesign
- patch reliability và patch selection

## 7.4. DROP

### Codec-direct / chroma-toxic

- `dct_mid_*`
- `cross_noise_ratio`
- `crlbp_*`
- `cblbp_*`
- `crsrm_*`

Lý do:

- utility nếu có thì phần lớn đến từ `JPEG/subsampling history`
- nuisance và collapse dưới cross-degradation quá lớn

### Current directional CFA axes

- `cfa_cr_pi_x`, `cfa_cr_pi_y`
- `cfa_cb_pi_x`, `cfa_cb_pi_y`
- `cfa_rg_pi_x`, `cfa_rg_pi_y`
- `cfa_bg_pi_x`, `cfa_bg_pi_y`

Lý do:

- trên `v4_exact`, nhiều feature trong nhóm này có `AUC_nat` ~ `0.70-0.80`
- utility thấp hơn rõ rệt so với `pi_xy`

Phán quyết:

- drop công thức hiện tại khỏi champion branch
- nếu nghiên cứu tiếp thì phải xem như một feature mới, không coi là carry-over

## 8. Các mệnh đề bất biến vẫn gây shortcut nhưng không thể xóa sạch hoặc không nên xóa

1. Unknown native JPEG / chroma history đã in vào pixel domain.
   - decode không xóa được
   - một transform tất định universal cũng không xóa chính xác được

2. Resize history của ảnh web có thể phá hủy trace vật lý trước khi ảnh đến pipeline.
   - preprocessing v4 không làm hại thêm
   - nhưng cũng không thể phục hồi CFA / SPN đã bị mất

3. Absence of prior compression trong AI PNG là bất đối xứng dataset-level.
   - không thể sửa trung thực bằng một bottleneck cố định

4. Image-formation traces thật như CFA / ISP / sensor noise không nên bị scrub mù quáng.

5. Patch aggregation không phải phép trung hòa.
   - nó có thể khuếch đại cả signal thật lẫn nuisance cục bộ

6. Degradation suite không được dùng như history equalizer mặc định.
   - nó chỉ hợp lệ ở vai trò audit, stress test, hoặc augmentation có điều kiện sau khi có telemetry deploy

## 9. Kế hoạch implementation cho phase tiếp theo

Notebook và code extractor sẽ được tách thành hai nhánh:

### Nhánh champion baseline

Chỉ implement trước:

- `safe_core`

Mục tiêu:

- tạo baseline sạch
- đạt mốc utility tối thiểu sau preprocessing v4
- làm control để so sánh mọi family mới

### Nhánh audit mở rộng

Implement tiếp theo theo thứ tự ưu tiên:

1. `CFA pi_xy redesign`
2. `Y-SRM`
3. `expanded NLF`
4. `chromatic aberration / PRNU / resampling`
5. `patch reliability studies`

## 10. Phán quyết cuối của `v0`

`feature_extract_v0` không được migrate nguyên 33 feature cũ.

Bản `v0` sau sửa đổi được khóa như sau:

- champion-safe baseline = `safe_core`
- `CFA pi_xy`, `Y-SRM`, `ps_alpha` = `priority-audit`
- `Y-LBP`, `expanded NLF`, `wavelet full`, `chromatic aberration`, `PRNU`, `resampling`, `patching` = `research-only`
- `dct_mid_*`, chroma LBP, chroma SRM, `cross_noise_ratio`, directional CFA hiện tại = `DROP`

Spec này là cơ sở để viết:

- notebook `02_feature_extraction_v4.ipynb`
- notebook audit `feature governance v4`
- benchmark `cross-degradation` cho model branch
