# Feature Extraction Standard v2

> Ghi chú trạng thái:
> Đây là bản đặc tả versioned chi tiết của taxonomy feature `v2`.
> Source-of-truth active hiện tại của pha feature extraction là
> [feature_extraction_standard.md](feature_extraction_standard.md).

## 1. Trạng thái

Tài liệu này là source-of-truth active cho pha feature extraction sau
[preprocessing_pipeline_standard_v4.md](preprocessing_pipeline_standard_v4.md).

Spec này được khóa bởi:

- [feature_spec_v2_validation.md](../reports/feature_spec_v2_validation.md)
- [summary.json](../../audit_output/studies/feature_spec_v2_validation_20260325/summary.json)
- [feature_set_metrics.csv](../../audit_output/studies/feature_spec_v2_validation_20260325/feature_set_metrics.csv)
- [single_feature_metrics.csv](../../audit_output/studies/feature_spec_v2_validation_20260325/single_feature_metrics.csv)
- [summary.json](../../audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/summary.json)
- [auc_sla_mapping.csv](../../audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/auc_sla_mapping.csv)
- [signal_nuisance_ratio.csv](../../audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/signal_nuisance_ratio.csv)

Spec này supersede:

- [feature_extraction_standard_v1.md](feature_extraction_standard_v1.md)
- [feature_extraction_standard_v0.md](feature_extraction_standard_v0.md)

## 2. Mục tiêu và phi mục tiêu

### 2.1. Mục tiêu

Pha này phải tạo ra một bảng đặc trưng đủ giàu để:

- giữ được phần evidence còn sống sau `v4_exact`
- tách `always-on` branch và `conditional` branch
- loại bỏ direct proxies
- chuẩn bị đầu vào cho downstream classifier có thể phi tuyến

Mục tiêu của pha này không phải là tìm một feature đơn "hoàn hảo".
Mục tiêu đúng là tạo một `multi-branch feature system` có thể:

- dùng feature bền hơn làm nền
- chỉ bật feature vi mô khi điều kiện vật lý còn cho phép
- để fusion layer học complementarity thay vì ép một feature đơn gánh cả bài toán

### 2.2. Phi mục tiêu

Spec này không cho phép:

- đưa degradation suite vào core preprocessing để "cân bằng history"
- admit feature chỉ vì clean AUC cao
- dùng nonlinear classifier để hợp thức hóa direct proxies
- frame bài toán như "phải tìm ra một feature không bị nén ảnh tác động"

## 3. Bối cảnh lý thuyết

### 3.1. Mô hình quan sát

Ta coi ảnh canonical sau preprocessing là:

`X = T(S, H_form, H_gen, H_codec)`

trong đó:

- `S`: scene / semantic content
- `H_form`: image-formation traces tự nhiên như demosaicing, CFA, sensor/ISP
- `H_gen`: generator traces của ảnh tổng hợp
- `H_codec`: lịch sử nén, subsampling, resize, transcode đã in vào pixel domain

Một feature `phi(X)` thường có dạng pha trộn:

`phi(X) = phi_sem(S) + phi_form(H_form) + phi_gen(H_gen) + phi_codec(H_codec) + eps`

Vấn đề của baseline DSP cũ không phải là "không có signal",
mà là `phi_codec` thường lớn hơn `phi_gen`.

### 3.2. Không có single perfect feature

Study frozen hiện tại xác nhận:

- `0` feature pass strict feature gate
- `0` feature-set pass champion gate

Điều này không có nghĩa "không còn feature để dùng".
Điều đúng là:

- yêu cầu một cue vừa vi mô, vừa codec-agnostic, vừa resize-robust là quá mạnh
- nhiều cue vật lý hợp lệ là `conditional`, không phải `always-on`
- champion readiness phải được xét ở mức branch/fusion/system, không phải mức feature đơn

### 3.3. Data processing inequality là guard, không phải veto

Với mọi operator lossy `G`, ta có:

`I(T_gen ; G(X)) <= I(T_gen ; X)`

Điều này không có nghĩa mọi phép chiếu lossy đều vô dụng.
Nó chỉ có nghĩa:

- không được overclaim rằng một phép lossy sẽ "tự động tạo thêm" generator trace
- mọi projection mới phải được chứng minh bằng audit utility/nuisance thật

Do đó spec này cho phép feature extraction dùng các transform lossy cục bộ như FFT, wavelet, residual, nhưng cấm diễn giải chúng như các phép "tẩy history" phổ quát.

### 3.4. Champion readiness không thể suy ra từ một ngưỡng AUC đơn

Diagnostics trong
[auc_sla_mapping.csv](../../audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/auc_sla_mapping.csv)
cho thấy:

| AUC | d' | TPR@FPR=5% |
|---|---:|---:|
| `0.7500` | `0.9539` | `0.2448` |
| `0.8322` | `1.3618` | `0.3885` |
| `0.8595` | `1.5245` | `0.4521` |
| `0.9335` | `2.1243` | `0.6842` |

Vì vậy:

- `AUC > 0.75` chỉ là ngưỡng "signal exists"
- nó không liên kết trực tiếp với SLA deploy
- mọi gate kiểu này chỉ có thể dùng ở mức research admission, không dùng làm champion verdict

## 4. Evidence thực nghiệm ràng buộc spec

Frozen study dùng giao của hai pipeline:

- `old_v1`
- `v4_exact`

Trên cùng `85615` ảnh accept bởi cả hai.

Sample frozen:

- clean LOGO: `1680`
- shift: `1344`
- nuisance real-only: `1176`
  - `4:4:4 = 588`
  - `4:2:0 = 588`

Kết quả hệ thống quan trọng nhất:

- `v4_exact` không làm feature xấu đi; nó làm lộ đúng utility và nuisance thật
- `0/0` pass hiện tại là bằng chứng chống `single perfect feature framing`
- current handcrafted DSP stack chưa đủ cho champion branch

Những mốc dữ liệu khóa taxonomy hiện tại:

| Family / set | Clean AUC | AUC_nat | Resize50 xdeg |
|---|---:|---:|---:|
| `control_minimal` | `0.5764` | `0.5194` | `0.5847` |
| `cfa_xy_only` | `0.8322` | `0.5394` | `0.5913` |
| `control_plus_cfa_xy` | `0.8595` | `0.6404` | `0.6472` |
| `wavelet_parent_only` | `0.6897` | `0.6505` | `0.6260` |
| `ysrm_only` | `0.7081` | `0.8192` | `0.5584` |
| `local_hetero_only` | `0.4845` | `0.5500` | `0.4690` |
| `crsrm_only` | `0.9335` | `0.9310` | `0.6136` |

Ý nghĩa:

- `control_minimal` là lower-bound control, không phải champion baseline
- `CFA` mạnh nhưng phải là conditional family
- wavelet có utility nhưng current formulation resize-fragile
- `Y-SRM` có signal nhưng current formulation quá nhiễm nuisance
- `crsrm` là toxic family dù clean AUC rất cao

## 5. Input contract

Feature extractor v2 chỉ nhận:

- `X_can_rgb8`
- shape `(248, 248, 3)`
- dtype `uint8`
- output từ preprocessing `v4_exact`

Không được dùng:

- source format
- source mode
- metadata / EXIF / ICC
- alpha flag / grayscale flag
- missingness cue
- reject-path cue

Derived views được phép:

- `RGB -> YCrCb`
- `R-G`, `B-G`
- residual maps trên `Y`
- FFT magnitude trên `Y`
- wavelet subbands trên `Y`
- class-independent validity / reliability masks

## 6. Nguyên tắc admission

### 6.1. Direct-proxy ban

Feature bị loại ngay nếu phụ thuộc trực tiếp vào:

- JPEG lattice direct statistics
- explicit chroma-bandwidth proxy
- source-format / source-mode cue
- missingness cue

### 6.2. Shift là diagnostic, không phải gate cứng

Diagnostics cho thấy:

- corr(mean shift, xdeg gap) ~= `0.097`
- corr(max shift, xdeg gap) ~= `-0.010`

Nên `z_shift` vẫn phải báo cáo, nhưng chỉ để:

- hiểu drift
- đánh giá calibration risk
- giải thích failure mode

không dùng làm hard veto độc lập.

### 6.3. Champion readiness ở mức set / branch / fusion

Một feature family chỉ được đưa vào champion stack khi:

- không vi phạm direct-proxy ban
- có utility sạch ở mức branch
- không làm fusion branch hỏng `AUC_nat` và `AUC_xdeg`
- được đánh giá với out-of-fold predictions và CI

Không có single-feature verdict nào đủ để gọi một family là champion-ready.

## 7. Metric bắt buộc

Mỗi feature, feature-set, branch, và fusion model phải báo cáo:

- `AUC_logo_clean`
- `AUC_nat = AUC(real 4:4:4 vs real 4:2:0)`
- `AUC_xdeg(g)` với:
  - `jpeg95_420`
  - `jpeg90_420`
  - `resize75_bilinear`
  - `resize50_bilinear`
  - `resize50_jpeg90_420`
- drift metrics:
  - `mean_abs_z_shift`
  - `median_abs_z_shift`
  - `max_abs_z_shift`

Ngoài ra:

- mọi study chính thức phải lưu out-of-fold predictions
- CI / bootstrap phải tính trên đúng frozen run đó
- report cuối phải có cả metrics branch-level lẫn fusion-level

## 8. Final taxonomy cho vòng triển khai kế tiếp

Spec này chốt một inventory triển khai gồm:

- `14` always-on features
- `4` conditional features
- `17` research-only features
- `1` class-independent validity score

Tổng cộng:

- `35` scalar features
- `1` validity score

### 8.1. Always-on branch A: `control_minimal`

Đây là lower-bound control branch.
Nó không phải champion branch.
Nó được giữ vì:

- là nhóm ít bẩn nhất trong stack legacy
- làm mốc so sánh cho mọi family mới
- có chi phí thấp và failure mode dễ hiểu

#### Inventory `control_minimal` (`8` features)

1. `frs_mid_variance`
2. `pearson_y_cr`
3. `pearson_y_cb`
4. `pearson_cr_cb`
5. `energy_ratio_chroma`
6. `spatial_snr_ratio`
7. `skew_noise_y`
8. `kurt_noise_y`

#### Những gì nhóm này đo

- `frs_mid_variance`
  - đo độ gồ ghề của phổ radial ở dải giữa
  - hiện là spectral cue sạch nhất trong stack legacy
  - single-feature trên `v4_exact`: clean `0.5496`, `AUC_nat = 0.5219`, max shift `0.3968`

- `pearson_*`, `energy_ratio_chroma`
  - đo coupling toàn cục giữa luma/chroma
  - không phải chroma microtexture, nên ít direct hơn nhiều feature Cr/Cb cục bộ

- `spatial_snr_ratio`
  - đo chênh lệch residual giữa vùng edge và vùng phẳng trên `Y`
  - phản ánh cách energy residual bám theo cấu trúc ảnh

- `skew_noise_y`, `kurt_noise_y`
  - đo hình dạng đuôi phân phối residual `Y`
  - hữu ích để bắt oversmooth hoặc residual bất thường

### 8.2. Always-on branch B: `fft_midband_y`

Đây là family mới được chọn chính thức cho v2.

#### Lý do chọn

- `DCT mid-band` bị loại vì dính trực tiếp vào `8x8 JPEG lattice`
- nhưng FFT radial mid-band không khóa vào lưới block như DCT
- low frequency bị scene semantics chi phối quá mạnh
- high frequency / Nyquist bị JPEG và resize phá mạnh nhất
- mid-band là vùng thỏa hiệp hợp lý nhất giữa utility và codec sensitivity

Spec này không overclaim rằng FFT mid-band "miễn nhiễm JPEG".
Phán quyết đúng là:

- nó ít codec-direct hơn DCT mid-band
- nó là hướng spectral nên ưu tiên hơn `ps_alpha`
- nó phải được audit như một family mới

#### Input và tiền xử lý

- input: `Y` channel chuẩn hóa về `float32`
- trừ mean trước FFT
- dùng cửa sổ Tukey nhẹ để giảm biên nếu cần, nhưng phải cố định
- chỉ dùng magnitude/radial statistics, không dùng phase

#### Vùng tần số mục tiêu

`fft_midband_y` phải đo trên dải giữa, không fit toàn phổ.
Implementation v2 dùng normalized radial support:

- `r in [0.12, 0.32]` cycles/pixel

và chia thành:

- inner-mid band
- outer-mid band

#### Inventory `fft_midband_y` (`6` features)

1. `fft_mid_logenergy`
2. `fft_mid_flatness`
3. `fft_mid_ring_var`
4. `fft_mid_inner_outer_ratio`
5. `fft_mid_anisotropy_hv`
6. `fft_mid_anisotropy_diag`

#### Ý nghĩa vật lý

- `fft_mid_logenergy`
  - tổng năng lượng log trong mid-band
  - bắt mức độ duy trì detail trung bình

- `fft_mid_flatness`
  - geometric mean / arithmetic mean của phổ mid-band
  - đo mức "spiky vs diffuse" của energy distribution

- `fft_mid_ring_var`
  - phương sai của profile radial trong dải giữa
  - mở rộng hợp lý từ `frs_mid_variance`

- `fft_mid_inner_outer_ratio`
  - tỷ lệ năng lượng giữa hai tiểu-dải
  - thay cho spectral slope toàn cục kiểu `ps_alpha`

- `fft_mid_anisotropy_hv`
  - chênh lệch năng lượng hướng ngang/dọc

- `fft_mid_anisotropy_diag`
  - chênh lệch năng lượng chéo

#### Expected failure mode

- vẫn nhạy với resize mạnh
- có thể bị scene texture tác động
- không được dùng như proxy cho JPEG quality

### 8.3. Conditional branch: `conditional_cfa_rgb`

`CFA` là conditional feature family chính thức của v2.

#### Phán quyết lý thuyết

Không tiếp tục target "làm CFA robust với resize50".
Lý do:

- CFA/demosaicing trace sống gần Nyquist
- resize low-pass có thể xóa hẳn bằng chứng đó
- nếu evidence đã mất, không có redesign nào phục hồi được thông tin đã mất

Do đó `CFA` phải là family `conditional`, không phải `always-on`.

#### Vì sao chọn RGB-domain thay vì Cr/Cb-domain

Frozen single-feature cho thấy:

- `cfa_rg_pi_xy`: clean `0.7242`, `AUC_nat = 0.5207`, max shift `0.6205`
- `cfa_cr_pi_xy`: clean `0.7553`, `AUC_nat = 0.5542`, max shift `1.0420`

Giải thích vật lý:

- `R-G`, `B-G` gần hơn với Bayer difference gốc
- `Cr`, `Cb` là tổ hợp tuyến tính của RGB
- sau chroma processing / subsampling, `Cr/Cb` bị low-pass khác với `R-G`
- vì vậy CFA dựa trên RGB-difference ổn định hơn với history codec

#### Inventory `conditional_cfa_rgb` (`4` features + validity)

1. `cfa_rg_pi_xy`
2. `cfa_bg_pi_xy`
3. `cfa_rgb_pi_xy_mean`
4. `cfa_rgb_pi_xy_gap`
5. `cfa_validity_score`

#### Ý nghĩa vật lý

- `pi_xy` đo năng lượng checkerboard `2x2` sau residual / high-pass phù hợp
- `mean` đo độ mạnh tổng thể của periodicity
- `gap` đo bất đối xứng giữa `R-G` và `B-G`
- `cfa_validity_score` đo xem bằng chứng high-frequency cần thiết cho CFA còn sống hay không

#### Quy định bắt buộc

- `cfa_validity_score` phải class-independent
- branch `CFA` chỉ được bật khi `cfa_validity_score` vượt ngưỡng đã audit
- report phải có:
  - coverage theo label
  - coverage theo generator
  - clean utility trên slice `M=1`
  - nuisance / xdeg trên slice `M=1`

### 8.4. Research-only branch A: `wavelet_decay`

#### Vì sao vẫn giữ hướng wavelet

Wavelet không bị loại vì "sai về lý thuyết".
Cái bị loại là formulation `parent-child correlation` tĩnh.

Frozen data:

- `wavelet_parent_only`: clean `0.6897`, `AUC_nat = 0.6505`, resize50 `0.6260`
- `wav_parent_corr_h`: max shift `1.8908`
- `wav_parent_corr_v`: max shift `1.9008`

Điều này cho thấy:

- utility có thật
- current formulation gãy mạnh khi scale thay đổi

Hướng đúng hơn là đo luật suy giảm năng lượng qua nhiều scale, thay vì tương quan tĩnh giữa hai level.

#### Inventory `wavelet_decay` (`6` features)

1. `wav_energy_l1`
2. `wav_energy_l2`
3. `wav_energy_l3`
4. `wav_decay_alpha`
5. `wav_ratio_l1_l2`
6. `wav_ratio_l2_l3`

#### Những gì family này đo

- năng lượng chi tiết qua các scale
- tốc độ suy giảm cross-scale
- deviation khỏi quy luật `1/f^alpha` gần đúng của natural image

#### Trạng thái

- phải extract trong v2 để audit
- không được đi vào champion branch mặc định

### 8.5. Research-only branch B: `dark_textured_hetero`

#### Vì sao giữ hướng này

Implementation hiện tại yếu:

- `local_hetero_only`: clean `0.4845`, `AUC_nat = 0.5500`

Nhưng lý do vật lý vẫn mạnh:

- ảnh camera thật có signal-dependent noise
- diffusion images thường đồng nhất hơn về residual/noise structure
- web JPEG phá mạnh vùng phẳng và vùng cạnh sắc
- vùng tối nhưng còn texture là nơi có khả năng giữ dị phương sai tốt hơn

#### Inventory `dark_textured_hetero` (`5` features)

1. `lochet_dark_flat_slope`
2. `lochet_dark_flat_r2`
3. `lochet_dark_flat_cv`
4. `lochet_dark_edge_flat_logratio`
5. `lochet_dark_monotone_violation`

#### Trạng thái

- phải extract để audit
- không được admit vào champion branch ở vòng đầu

### 8.6. Research-only branch C: `content_adaptive_y_srm`

#### Vì sao không drop hẳn `Y-SRM`

`Y-SRM` không phải zero-signal family.
Nhưng boundary masking hình học hiện tại đã thất bại:

- `ysrm_only`: clean `0.7081`, `AUC_nat = 0.8192`
- `ysrm_native_mask_only`: clean `0.6900`, `AUC_nat = 0.8522`
- `ysrm_union_mask_only`: clean `0.6786`, `AUC_nat = 0.8547`

Vấn đề:

- JPEG artifact không chỉ nằm ở pixel biên block
- masking biên thô làm mất phần signal nhưng không bỏ được ringing

Do đó hướng còn lại duy nhất hợp lệ là `content-adaptive Y-SRM`.

#### Inventory `content_adaptive_y_srm` (`6` features)

1. `ysrm_midtex_edge3_energy`
2. `ysrm_midtex_edge3_mar`
3. `ysrm_midtex_square3_energy`
4. `ysrm_midtex_square3_mar`
5. `ysrm_midtex_square5_energy`
6. `ysrm_midtex_square5_mar`

#### Quy định

- chỉ tính trên `medium-texture mask`
- loại bỏ flat regions
- loại bỏ strong-edge regions
- dùng estimator robust hơn mean nếu cần

#### Trạng thái

- extract để audit
- không admit vào champion branch mặc định

## 9. Backlog nghiên cứu không thuộc inventory v2

Những family sau có cơ sở khoa học nhưng chưa được đưa vào inventory v2:

### 9.1. `expanded_nlf`

Giữ ở backlog vì:

- current 5-feature NLF quá yếu
- nhưng family NLF chưa bị bác về mặt vật lý

### 9.2. `chromatic_aberration`

Lý do:

- là image-formation trace thật
- có cơ sở tốt trong pháp y ảnh
- nhưng có nguy cơ quá yếu trên patch `248x248`

### 9.3. `prnu_spn`

Lý do:

- nền tảng pháp y rất mạnh
- nhưng ảnh web và patch nhỏ có thể làm SPN quá yếu

### 9.4. `resampling_traces`

Lý do:

- có giá trị cho hậu kỳ
- nhưng dễ học post-process history hơn generator trace

### 9.5. `patch_reliability`

Lý do:

- có thể là chìa khóa cho conditional fusion
- nhưng nó là tầng gating / selection nhiều hơn là một feature family thuần

## 10. Family bị loại chính thức

Những family sau bị drop ở mức spec, không re-admit trừ khi có formulation mới:

- `dct_mid_*`
- chroma SRM
- chroma LBP
- `cross_noise_ratio`
- directional `CFA pi_x / pi_y`

### 10.1. Giải thích `cross_noise_ratio`

`cross_noise_ratio` vẫn `DROP`.
Lý do đúng là:

- đây là direct chroma-band proxy
- raw nuisance pooled AUC của nó thực ra khoảng `0.7732`
- con số LOGO `< 0.5` trong report cũ là artifact của logistic trên feature heavy-tailed

Vì vậy verdict `DROP` là vì cơ chế, không phải vì sign convention.

### 10.2. Vì sao `ps_alpha` không ở inventory v2

`ps_alpha` có utility có thật, nhưng:

- whole-spectrum slope quá resize-fragile
- hiện không sạch bằng hướng `fft_midband_y`
- dễ bị hiểu nhầm như một summary proxy cho compression history

Do đó:

- `ps_alpha` không bị cấm về mặt lý thuyết
- nhưng không còn là family ưu tiên cho implementation v2

## 11. Tài liệu nền tảng khoa học

Những hướng trong spec này dựa trên các trục lý thuyết sau:

- demosaicing / CFA periodicity
  - Bayram, Sencar, Memon, 2008
  - [DOI](https://doi.org/10.1016/j.diin.2008.06.004)
- wavelet joint statistics
  - Portilla, Simoncelli, 2000
  - [Abstract](https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html)
- chromatic aberration
  - Johnson, Farid, 2006
  - [Paper](https://farid.berkeley.edu/downloads/publications/acm06c.pdf)
- sensor pattern noise / PRNU
  - Lukás, Fridrich, Goljan, 2006
  - [IEEE page](https://ieeexplore.ieee.org/document/1634362)
- resampling traces
  - Popescu, Farid, 2005
  - [IEEE page](https://ieeexplore.ieee.org/document/1381775)
- generalized noise model
  - Thai, Retraint, Cogranne, 2015
  - [ScienceDirect page](https://www.sciencedirect.com/science/article/abs/pii/S1051200415003012)
- patch reliability / selection
  - Liu et al., 2021
  - [MDPI paper](https://www.mdpi.com/1424-8220/21/14/4701)
- rich residual models
  - Fridrich, Kodovsky, 2012
  - [PDF](https://dde.binghamton.edu/vholub/pdf/TIFS2012-SRM.pdf)
- frequency-based fake detection caveats
  - Wang et al., 2019
  - [arXiv](https://arxiv.org/abs/1912.11035)
  - Chandrasegaran et al., 2021
  - [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chandrasegaran_A_Closer_Look_at_Fourier_Spectrum_Discrepancies_for_CNN-Generated_Images_CVPR_2021_paper.pdf)
  - Dong et al., 2022
  - [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Think_Twice_Before_Detecting_GAN-Generated_Fake_Images_From_Their_Spectral_CVPR_2022_paper.pdf)

## 12. Thiết kế downstream sau feature extraction

Downstream model không còn bị ràng buộc vào flat logistic regression.
Nhưng spec này khóa 4 nguyên tắc:

1. `always-on` và `conditional` phải là hai nhánh tách biệt.
2. validity gate phải class-independent.
3. nonlinear fusion chỉ được dùng sau khi drop direct proxies.
4. calibration phải báo cáo ở mức branch và system.

Hướng ưu tiên:

1. `always-on fusion` trên:
   - `control_minimal`
   - `fft_midband_y`
2. `conditional CFA branch`
3. `research branches` chỉ để audit
4. fusion layer:
   - logistic baseline
   - LightGBM / boosted tree
   - gated mixture với calibration

## 13. Kế hoạch thực thi ngay sau spec này

### 13.1. Code

1. viết lại `src/feature_extraction` theo inventory v2
2. implement các family:
   - `control_minimal`
   - `fft_midband_y`
   - `conditional_cfa_rgb`
   - `wavelet_decay`
   - `dark_textured_hetero`
   - `content_adaptive_y_srm`

### 13.2. Notebook

1. viết lại `02_feature_extraction_v4.ipynb`
2. notebook chỉ gọi API trong `src/feature_extraction`
3. output:
   - feature table v2
   - branch-level QC
   - validity coverage report

### 13.3. Validation

1. lưu out-of-fold predictions cho từng branch
2. report lại:
   - branch AUC
   - fusion AUC
   - natural nuisance
   - cross-degradation
   - CI trên exact frozen run

## 14. Mệnh đề khóa cuối

1. `v4_exact` giữ nguyên làm preprocessing source-of-truth.
2. Không dùng degradation suite làm canonical preprocessing.
3. Không tiếp tục săn "single perfect feature".
4. Current flat handcrafted DSP stack chưa đủ cho champion branch.
5. Hướng đúng là `always-on + conditional + research branches + nonlinear fusion`.
