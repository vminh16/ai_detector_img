# Feature Extraction Standard v1

## 1. Trạng thái

Tài liệu này là source-of-truth active cho pha feature extraction sau [preprocessing_pipeline_standard_v4.md](docs/specs/preprocessing_pipeline_standard_v4.md).

Spec này được khóa bởi:

- [feature_spec_v2_validation.md](docs/reports/feature_spec_v2_validation.md)
- [feature_space_update.md](docs/reports/feature_space_update.md)
- [shortcut_risk_validation.md](docs/reports/shortcut_risk_validation.md)
- [feature_spec_v2_validation_20260325](audit_output/studies/feature_spec_v2_validation_20260325)

Spec này supersede [feature_extraction_standard_v0.md](docs/specs/feature_extraction_standard_v0.md).

## 2. Mục tiêu pha này

Mục tiêu của pha feature extraction không phải tìm một feature đơn "hoàn hảo".
Mục tiêu đúng là:

- chọn một tập feature có complementarity
- tách feature `always-on` và feature `conditional`
- loại direct proxies
- cung cấp đầu vào cho một downstream classifier có thể phi tuyến nếu cần

## 3. Input contract

Feature extractor v1 chỉ được nhận:

- `X_can_rgb8`
- shape `(248, 248, 3)`
- dtype `uint8`
- đã qua preprocessing v4 exact-crop

Không được dùng:

- source format
- source mode
- metadata / EXIF / ICC
- alpha flag / grayscale flag
- missingness cue
- reject-path cue

Derived views được phép:

- `RGB -> YCrCb`
- residual maps
- FFT / radial statistics
- wavelet maps
- RGB-difference maps
- reliability masks

## 4. Nguyên tắc admissibility mới

### 4.1. Direct-proxy ban

Mọi feature có phụ thuộc trực tiếp vào forbidden variable bị loại ngay, bất kể clean AUC đẹp đến đâu.

Forbidden dependencies:

- JPEG lattice direct statistics
- explicit chroma-bandwidth proxies
- source-format / source-mode cues
- missingness cues

Direct-proxy drop:

- `dct_mid_*`
- `cross_noise_ratio`
- chroma SRM
- chroma LBP
- directional `CFA pi_x / pi_y`

### 4.2. Không còn hard single-feature gate

Spec này hủy bỏ hard gate kiểu:

- `single feature clean_auc > 0.75`
- `single feature resize50_max_shift < 1.0`

Lý do:

- gate đó không liên kết với SLA
- quá khắt khe với forensic cues vi mô
- và không phân biệt `always-on` với `conditional` families

### 4.3. Shift là diagnostic, không phải hard veto

Feature/set shift vẫn phải báo cáo, nhưng chỉ dùng cho:

- drift diagnosis
- calibration risk
- giải thích vì sao `xdeg` thay đổi

Shift không được dùng làm hard veto một mình.

### 4.4. Champion readiness chỉ được xét ở system level

Không có feature hoặc feature-set nào được gọi là champion-ready chỉ vì clean AUC cao.

Champion readiness chỉ được xét sau khi có:

- branch models
- validity gates
- fusion model
- system-level metrics gắn với SLA

## 5. Benchmark protocol bắt buộc

Mỗi feature family / feature set / branch phải báo cáo:

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

Ngoài ra, mỗi study chính thức phải:

- lưu out-of-fold predictions
- báo cáo CI / bootstrap trên exact frozen run

## 6. Taxonomy v1

### 6.1. Always-on baseline

#### `control_minimal`

Đây là lower-bound branch để:

- đo signal còn lại ít shortcut nhất
- làm control cho mỗi branch mới

Thành phần:

- `frs_mid_variance`
- `pearson_y_cr`
- `pearson_y_cb`
- `pearson_cr_cb`
- `energy_ratio_chroma`
- `spatial_snr_ratio`
- `skew_noise_y`
- `kurt_noise_y`

Quy định:

- được implement trước
- không được gọi là champion branch

### 6.2. Conditional feature families

Những family này được phép đi tiếp, nhưng chỉ với validity detector class-independent.

#### `CFA pi_xy`

Current evidence:

- clean mạnh
- `AUC_nat` khá ổn
- sụp mạnh dưới resize/JPEG

Quy định:

- không tiếp tục frame như feature cần "robust với resize50"
- chỉ được dùng trong `conditional CFA branch`
- ưu tiên `RGB-difference CFA` hơn `Cr/Cb CFA`

#### `Wavelet decay-law branch`

Current `wav_parent_corr` implementation không admissible.
Family wavelet vẫn được giữ, nhưng phải đổi formulation sang:

- multiscale energy decay
- adjacent-level ratios
- `1/f^alpha` normalization

Quy định:

- không port current `wav_parent_corr_h/v` nguyên trạng vào champion branch
- chỉ implement lại trong `conditional / research bridge branch`

### 6.3. Research-only families

#### `Content-adaptive Y-SRM`

Current boundary masking đã thất bại.
Hướng tiếp theo chỉ được là:

- content-adaptive
- medium-texture
- robust estimator

#### `Dark-textured heteroskedasticity`

Current implementation quá yếu.
Nhưng hướng vật lý vẫn có cơ sở.

#### `ps_alpha`

Utility có thật nhưng resize-fragile và không đủ sạch để admit.

#### `Expanded NLF`

Chưa đủ utility ở implementation hiện tại.

#### Backlog nghiên cứu

- patch reliability / patch selection
- chromatic aberration
- PRNU / sensor pattern noise
- resampling traces

### 6.4. Drop

- `dct_mid_*`
- chroma SRM
- chroma LBP
- `cross_noise_ratio`
- directional `CFA pi_x / pi_y`

## 7. Thiết kế downstream model

### 7.1. Flat logistic regression không còn là assumption bắt buộc

Downstream model được phép phi tuyến.
Nhưng nonlinear model chỉ được dùng sau khi feature taxonomy đã được audit.

### 7.2. Architecture ưu tiên

Hướng ưu tiên cho phase tiếp theo:

1. `always-on baseline branch`
2. `conditional CFA branch`
3. `future wavelet-decay branch`
4. `future heteroskedasticity branch`
5. `fusion layer`

Fusion layer có thể là:

- LightGBM / boosted tree
- MLP nhỏ có calibration
- gated mixture of experts

### 7.3. Điều kiện bắt buộc cho conditional branches

Mỗi conditional branch phải báo cáo thêm:

- validity mask / reliability score
- coverage theo label và generator
- clean utility trên slice `M=1`
- nuisance utility trên slice `M=1`
- invalid-slice neutrality

## 8. Kế hoạch implementation

### 8.1. Pha code gần nhất

1. implement `control_minimal` ở `src/feature_extraction`
2. tách branch audit cho:
   - `RGB-domain CFA pi_xy`
   - wavelet decay-law
   - dark-textured heteroskedasticity

### 8.2. Pha benchmark

1. lưu out-of-fold predictions cho mỗi branch
2. báo cáo CI cho delta so với control baseline
3. đối sánh linear fusion vs nonlinear fusion

## 9. Mệnh đề hệ thống cần giữ nguyên

1. `v4_exact` giữ nguyên làm preprocessing source-of-truth.
2. không dùng degradation suite làm canonical preprocessing.
3. không train champion model trên flat feature stack hiện tại.
4. không tiếp tục săn "single perfect feature".
