# Validation `feature_spec_v2` trên `old_v1` vs `v4_exact`

## 1. Trạng thái và phạm vi

Báo cáo này là report active hiện tại cho pha feature extraction sau [preprocessing_pipeline_standard_v4.md](docs/specs/preprocessing_pipeline_standard_v4.md).

Mục tiêu của report này không phải tìm ra "single perfect feature".
Mục tiêu đúng là:

- xác định family nào là `always-on`
- family nào là `conditional`
- family nào là `research-only`
- family nào là `drop`

và chốt lại xem current handcrafted DSP stack còn có đủ headroom để đi đến champion branch hay không.

Báo cáo này supersede các kết luận diễn giải cũ trong:

- [feature_spec_v1_validation.md](docs/reports/feature_spec_v1_validation.md)
- [feature_spec_v0_review.md](docs/reports/feature_spec_v0_review.md)

## 2. Nguồn bằng chứng

### 2.1. Study chính

Script:

- [feature_spec_v2_validation.py](script/studies/feature_spec_v2_validation.py)

Artifact:

- [summary.json](audit_output/studies/feature_spec_v2_validation_20260325/summary.json)
- [feature_set_metrics.csv](audit_output/studies/feature_spec_v2_validation_20260325/feature_set_metrics.csv)
- [single_feature_metrics.csv](audit_output/studies/feature_spec_v2_validation_20260325/single_feature_metrics.csv)
- [feature_shift_metrics.csv](audit_output/studies/feature_spec_v2_validation_20260325/feature_shift_metrics.csv)
- [feature_set_shift_metrics.csv](audit_output/studies/feature_spec_v2_validation_20260325/feature_set_shift_metrics.csv)
- [feature_gate_summary.csv](audit_output/studies/feature_spec_v2_validation_20260325/feature_gate_summary.csv)
- [feature_set_gate_summary.csv](audit_output/studies/feature_spec_v2_validation_20260325/feature_set_gate_summary.csv)

### 2.2. Diagnostics bổ sung

Script:

- [feature_spec_v2_diagnostics.py](script/studies/feature_spec_v2_diagnostics.py)

Artifact:

- [summary.json](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/summary.json)
- [auc_sla_mapping.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/auc_sla_mapping.csv)
- [resize_shift_redundancy.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/resize_shift_redundancy.csv)
- [control_minimal_generalization.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/control_minimal_generalization.csv)
- [control_minimal_correlation_matrix.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/control_minimal_correlation_matrix.csv)
- [cross_noise_ratio_pooled.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/cross_noise_ratio_pooled.csv)
- [cross_noise_ratio_by_generator.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/cross_noise_ratio_by_generator.csv)
- [logo_bootstrap_comparisons.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/logo_bootstrap_comparisons.csv)
- [signal_nuisance_ratio.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/signal_nuisance_ratio.csv)

### 2.3. Dữ liệu và sample

Frozen study được chạy trên giao của hai manifest:

- [manifest.csv](data/processed/manifest.csv)
- [manifest.csv](data/processed_v4_rgb248_r4_exact/manifest.csv)

Tổng số row chung được accept bởi cả hai pipeline:

- `85615`

Mẫu frozen:

- clean LOGO: `1680`
- feature shift: `1344`
- real-only nuisance: `1176`
  - `4:4:4 = 588`
  - `4:2:0 = 588`

## 3. Những điều đã được kiểm chứng lại

### 3.1. Gate `AUC > 0.75` không liên quan tới SLA

Đây chỉ là ngưỡng "minimum signal exists", không phải champion threshold.
Diagnostics bổ sung trong [auc_sla_mapping.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/auc_sla_mapping.csv) cho thấy:

| AUC | d' | TPR@FPR=5% (Gaussian equal-variance) |
|---|---:|---:|
| `0.7500` | `0.9539` | `0.2448` |
| `0.8322` | `1.3618` | `0.3885` |
| `0.8595` | `1.5245` | `0.4521` |
| `0.9335` | `2.1243` | `0.6842` |

Hệ quả:

- `AUC > 0.75` không được diễn giải thành champion-readiness
- current best handcrafted set trong frozen study vẫn cách xa SLA

### 3.2. Hard gate `resize50_max_shift < 1.0` không nên còn là gate admission

Diagnostics bổ sung trong [summary.json](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/summary.json) cho thấy:

- corr(mean shift, xdeg gap) ~= `0.097`
- corr(max shift, xdeg gap) ~= `-0.010`

Tức là shift và xdeg không đồng nhất với nhau.
Shift vẫn có giá trị:

- diagnostic drift
- calibration risk
- phá hình phân phối

Nhưng nó không nên là hard gate nhị phân cho feature admission.

### 3.3. `v4_exact` vẫn là preprocessing core đúng

Frozen study xác nhận thêm một lần nữa:

- `v4_exact` không tạo thêm shortcut
- `v4_exact` không "làm feature xấu đi"
- `v4_exact` phơi bày utility thật và nuisance thật

Ví dụ:

| Feature set | old_v1 clean | v4_exact clean | old_v1 nat | v4_exact nat |
|---|---:|---:|---:|---:|
| `crsrm_only` | `0.6827` | `0.9335` | `0.8980` | `0.9310` |
| `cfa_xy_only` | `0.5029` | `0.8322` | `0.4905` | `0.5394` |

Blocker hiện tại không nằm ở preprocessing.
Blocker nằm ở:

- feature formulation
- family coupling với nuisance
- và architecture của downstream classifier

## 4. Không có feature "hoàn hảo", và đó là điều bình thường

Kết quả frozen:

- `0` feature pass feature-level strict gate
- `0` feature-set pass champion gate

Điều này không có nghĩa là "không còn feature để dùng".
Nó có nghĩa:

- hard single-feature gate đang quá mạnh đối với forensic cues vi mô
- bài toán đúng là bài toán `multi-feature + conditional validity`

Nói cách khác:

- tìm một feature đơn vừa micro, vừa codec-agnostic, vừa resize-robust là yêu cầu quá mức
- multi-feature fusion và conditional branches là hướng đúng hơn về mặt toán học và vật lý

## 5. Đánh giá lại các family hiện tại

### 5.1. `control_minimal` chỉ là lower-bound control

Frozen metric trên `v4_exact`:

- clean `0.5764`
- `AUC_nat = 0.5194`
- `resize50_xdeg = 0.5847`

Diagnostics bổ sung trong [control_minimal_generalization.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/control_minimal_generalization.csv):

- pooled random CV `0.6019`
- LOGO `0.5764`
- within-generator CV: `0.5691` đến `0.6975`

Và ma trận tương quan trong [control_minimal_correlation_matrix.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/control_minimal_correlation_matrix.csv) cho thấy một vài cặp có tương quan vừa phải/mạnh (`pearson_y_cr` vs `pearson_y_cb`, `skew_noise_y` vs `kurt_noise_y`), nhưng không có bằng chứng rằng kết quả `0.5764` chủ yếu do feature redundancy.

Phán quyết:

- `control_minimal` là control baseline
- không phải champion baseline

### 5.2. `CFA pi_xy` phải chuyển từ "robust redesign" sang `conditional feature family`

Frozen metric trên `v4_exact`:

- `cfa_xy_only`: clean `0.8322`, `AUC_nat = 0.5394`, `resize50_xdeg = 0.5913`
- `control_plus_cfa_xy`: clean `0.8595`, `AUC_nat = 0.6404`, `resize50_xdeg = 0.6472`

Single-feature:

- `cfa_cr_pi_xy`: clean `0.7553`, `AUC_nat = 0.5542`, max shift `1.0420`
- `cfa_rg_pi_xy`: clean `0.7242`, `AUC_nat = 0.5207`, max shift `0.6205`

Diễn giải vật lý đúng:

- `R-G` gần hơn với Bayer difference gốc
- `Cr` là tổ hợp tuyến tính của RGB nên nhạy hơn với chroma processing / subsampling

Phán quyết mới:

- không tiếp tục frame `CFA` như một feature cần "làm cho robust với resize50"
- nếu Nyquist-scale demosaicing trace đã bị low-pass cắt bởi resize, không có redesign nào phục hồi được thông tin đã mất
- `CFA` phải được xem như `conditional feature family`

Ý nghĩa kỹ thuật:

- cần một validity detector class-independent
- chỉ bật `CFA branch` khi bằng chứng cho thấy high-frequency evidence còn tồn tại

### 5.3. `Wavelet parent corr` hiện tại không admissible, nhưng hướng wavelet chưa đóng lại

Frozen metric trên `v4_exact`:

- `wavelet_parent_only`: clean `0.6897`, `AUC_nat = 0.6505`, `resize50_xdeg = 0.6260`
- `wav_parent_corr_h`: max shift `1.8908`
- `wav_parent_corr_v`: max shift `1.9008`

Phán quyết mới:

- current `parent-child correlation` không được xếp `always-on`
- nhưng hướng wavelet vẫn còn giá trị nếu đổi formulation

Hướng đúng hơn:

- wavelet energy decay law qua nhiều level
- adjacent-level energy ratios
- normalization theo `1/f^alpha`

Đây là family mới, không phải minor patch của feature cũ.

### 5.4. `Y-SRM` không nên bị bỏ hẳn, nhưng boundary masking hiện tại đã thất bại

Frozen metric trên `v4_exact`:

- `ysrm_only`: clean `0.7081`, `AUC_nat = 0.8192`
- `ysrm_native_mask_only`: clean `0.6900`, `AUC_nat = 0.8522`
- `ysrm_union_mask_only`: clean `0.6786`, `AUC_nat = 0.8547`

Kết luận:

- geometric block-boundary masking không giải được nuisance
- masking hiện tại có thể còn làm tăng nuisance vì tập trung vào flat regions codec-sensitive

Phán quyết mới:

- `Y-SRM` ở lại `research-only`
- hướng tiếp theo phải là `content-adaptive Y-SRM`
  - medium-texture regions
  - bỏ qua flat và strong-edge regions
  - robust estimator thay vì simple mean

### 5.5. `Local heteroskedasticity` hiện tại yếu, nhưng vẫn là hướng đáng nghiên cứu

Frozen metric trên `v4_exact`:

- `local_hetero_only`: clean `0.4845`, `AUC_nat = 0.5500`
- `control_plus_local_hetero`: clean `0.5650`, `AUC_nat = 0.5150`

Điều này cho thấy implementation hiện tại quá yếu.
Nhưng phân tích vật lý vẫn có cơ sở:

- diffusion images có khuynh hướng homoskedastic hơn
- ảnh chụp thật có signal-dependent noise

Phán quyết:

- không promote current implementation
- nhưng giữ hướng `dark-textured heteroskedasticity` ở mức ưu tiên nghiên cứu cao

### 5.6. `cross_noise_ratio` vẫn phải `DROP`, nhưng lý do trong report cũ chưa đúng

Frozen report cũ diễn giải `AUC_nat = 0.3821` như một kết quả nuisance nghịch dấu.
Diagnostics bổ sung trong:

- [cross_noise_ratio_pooled.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/cross_noise_ratio_pooled.csv)
- [cross_noise_ratio_by_generator.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/cross_noise_ratio_by_generator.csv)

cho thấy:

- raw pooled AUC của `cross_noise_ratio` trong nuisance task là `0.7732`
- raw per-generator AUC nằm trong khoảng `0.6813` đến `0.8233`
- median `cross_noise_ratio` của `4:2:0` cao hơn `4:4:4`

Hệ quả:

- con số `0.3821` trong single-feature LOGO do logistic trên feature heavy-tailed, không scale/log-transform, dẫn đến hệ số gần 0 ở đa số fold
- do đó, `0.3821` không nên được diễn giải như bằng chứng vật lý cho "sign inversion"

Tuy nhiên verdict `DROP` vẫn giữ nguyên, vì:

- `cross_noise_ratio` là direct chroma-band proxy
- nó vi phạm direct-proxy ban ngay từ cơ chế

### 5.7. Chance-corrected signal/nuisance ratio bổ sung góc nhìn đúng hơn

Diagnostics bổ sung trong [signal_nuisance_ratio.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/signal_nuisance_ratio.csv) dùng:

`rho = |AUC_clean - 0.5| / |AUC_nat - 0.5|`

Kết quả:

| Feature set | Preprocess | rho |
|---|---|---:|
| `crsrm_only` | `old_v1` | `0.459` |
| `crsrm_only` | `v4_exact` | `1.006` |
| `cfa_xy_only` | `old_v1` | `0.306` |
| `cfa_xy_only` | `v4_exact` | `8.441` |

Phân tích này quan trọng hơn việc chỉ nhìn clean AUC:

- `v4_exact` cải thiện utility-vs-nuisance cho `CFA`
- nhưng gần như không biến `crsrm` thành family sạch

## 6. Về CI và p-value

Report cũ đúng là còn thiếu CI.
Diagnostics bổ sung có [logo_bootstrap_comparisons.csv](audit_output/studies/feature_spec_v2_validation_20260325/diagnostics/logo_bootstrap_comparisons.csv), nhưng artifact này được tạo từ một fresh recompute, không phải từ out-of-fold predictions frozen của study chính.

Vì vậy:

- artifact bootstrap này chỉ được xem là exploratory
- chưa được đưa vào source-of-truth để chốt tier feature

Requirement mới:

- mỗi study sau phải lưu out-of-fold predictions của exact frozen run
- bootstrap/CI phải được tính trên artifact đó

## 7. Từ report sang chiến lược thiết kế mới

### 7.1. Không cần single perfect feature

Thiết kế đúng không phải:

- một feature đơn vừa vi mô, vừa resize-robust, vừa codec-agnostic

Thiết kế đúng là:

- `always-on` families
- `conditional` families
- multi-feature fusion
- nonlinear downstream model nếu cần

### 7.2. Nhưng nonlinear model không được dùng để "che" shortcut

Nonlinear model được cho phép, nhưng chỉ sau khi:

- direct-proxy families đã bị drop
- validity gate là class-independent
- benchmarking có báo cáo per-branch và system-level

Nói cách khác:

- nonlinear fusion là công cụ kết hợp feature
- không phải cách để hợp thức hóa feature bẩn

### 7.3. Conditional feature set là giải pháp hợp lý nhất hiện tại

Current evidence ủng hộ system hai tầng:

1. `always-on branch`
   - control-like, low-risk, utility không cao nhưng bền hơn
2. `conditional branches`
   - `CFA branch` chỉ bật khi có Nyquist-survival evidence
   - future `wavelet-decay branch`
   - future `dark-textured heteroskedasticity branch`
3. `fusion layer`
   - nonlinear / degradation-aware
   - có calibration riêng

## 8. Literature-guided feature directions

Những hướng dưới đây có cơ sở từ khoa học pháp y ảnh và nên đưa vào backlog feature:

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
- patch selection / reliability
  - Liu et al., 2021
  - [MDPI paper](https://www.mdpi.com/1424-8220/21/14/4701)
- rich residual models
  - Fridrich, Kodovsky, 2012
  - [PDF](https://dde.binghamton.edu/vholub/pdf/TIFS2012-SRM.pdf)
- frequency-based fake detection caveat
  - Wang et al., 2019
  - [arXiv](https://arxiv.org/abs/1912.11035)
  - Chandrasegaran et al., 2021
  - [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chandrasegaran_A_Closer_Look_at_Fourier_Spectrum_Discrepancies_for_CNN-Generated_Images_CVPR_2021_paper.pdf)
  - Dong et al., 2022
  - [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Think_Twice_Before_Detecting_GAN-Generated_Fake_Images_From_Their_Spectral_CVPR_2022_paper.pdf)

## 9. Phán quyết cuối cùng

1. Current handcrafted DSP families trong repo chưa đủ để làm champion model.
2. `0 feature pass / 0 set pass` là bằng chứng rằng current flat-feature framing đã chạm trần.
3. Điều đó không có nghĩa "không còn feature để dùng".
4. Nghĩa đúng là:
   - bỏ tư duy single perfect feature
   - chuyển sang `always-on + conditional branches + nonlinear fusion`
5. `v4_exact` giữ nguyên làm preprocessing source-of-truth.
