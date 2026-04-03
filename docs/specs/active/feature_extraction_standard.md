# Đặc tả trích chọn đặc trưng hiện hành

## 1. Trạng thái và phạm vi

Đây là đặc tả active cho pha feature extraction của nhánh `codex/preprocessing-v4-core`.

Phiên bản triển khai hiện tại:

- **mã nguồn**: [src/feature_extraction](C:/Users/USER/Desktop/ai_detector_img/src/feature_extraction)
- **quy ước artifact**: `v2_rgb248_exact_multibranch`
- **notebook orchestration**: [02_feature_extraction.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/02_feature_extraction.ipynb)

Tài liệu này hợp nhất nội dung thiết kế active từ:

- [feature_extraction_standard_v2.md](../archive/feature_extraction_standard_v2.md)
- [feature_space_validation_summary.md](../../reports/active/feature_space_validation_summary.md)
- [training_baseline_validation.md](../../reports/active/training_baseline_validation.md)

## 2. Mục tiêu của pha này

Pha feature extraction phải làm ba việc cùng lúc:

1. giữ lại càng nhiều signal thật về khác biệt giữa ảnh diffusion và ảnh real càng tốt,
2. tránh đưa direct proxy hoặc cue codec quá lộ liễu vào downstream model,
3. tách rõ các cue:
   - luôn dùng được,
   - chỉ dùng khi có điều kiện vật lý phù hợp,
   - và chỉ nên dùng cho nghiên cứu.

Pha này **không** nhằm tìm một feature đơn “miễn nhiễm nén”.
Mục tiêu đúng là xây một **hệ đặc trưng nhiều nhánh** có thể:

- đặt feature sạch hơn ở nhánh nền,
- dùng feature vi mô dưới chế độ `conditional`,
- và cô lập các branch chưa đủ sạch ở trạng thái `research-only`.

## 3. Mô hình quan sát và lý do không thể có feature hoàn hảo

Ta coi patch canonical đầu vào là:

`X = T(S, H_form, H_gen, H_codec)`

Trong đó:

- `S`: semantic content của cảnh
- `H_form`: dấu vết image formation tự nhiên như demosaicing, CFA, sensor/ISP
- `H_gen`: dấu vết do generator tạo ra
- `H_codec`: lịch sử nén, subsampling, resize, transcode đã in vào pixel

Một feature bất kỳ thường có dạng:

`phi(X) = phi_sem(S) + phi_form(H_form) + phi_gen(H_gen) + phi_codec(H_codec) + eps`

Trên dữ liệu web, bài toán khó nằm ở chỗ:

- `phi_codec` thường mạnh ngang hoặc mạnh hơn `phi_gen`

Nên nếu chỉ nhìn clean AUC:

- ta rất dễ chọn nhầm một feature “mạnh” nhưng thực ra độc.

## 4. Input contract

Pha này chỉ nhận:

- `X_can_rgb8`
- shape `(248, 248, 3)`
- dtype `uint8`
- output đúng từ preprocessing active

Derived views được phép:

- `RGB -> YCrCb`
- `R-G`, `B-G`
- residual trên `Y`
- FFT magnitude trên `Y`
- wavelet subbands trên `Y`
- validity mask class-independent

Không được dùng:

- format gốc
- mode gốc
- metadata
- alpha flag
- grayscale flag
- missingness cue

## 5. Các định nghĩa đánh giá bắt buộc

### 5.1. `AUC_logo_clean`

AUC trên benchmark clean với giao thức leave-one-generator-out hoặc split clean tương đương.

Ý nghĩa:

- đo utility tổng quát hóa trên các generator ngoài train.

### 5.2. `AUC_nat`

`AUC_nat = AUC(real 4:4:4 vs real 4:2:0)`

Ý nghĩa:

- đo mức độ feature/model đang đọc natural subsampling history trong dữ liệu real-only.

Nếu `AUC_nat` cao:

- feature/model có nguy cơ mạnh đang dùng codec history thay vì generator trace.

### 5.3. `AUC_xdeg(g)`

Là AUC của model khi:

- train trên dữ liệu clean,
- test trên dữ liệu đã qua một degradation `g`.

Các degradation bắt buộc:

- `jpeg95_420`
- `jpeg90_420`
- `resize75_bilinear`
- `resize50_bilinear`
- `resize50_jpeg90_420`

Ý nghĩa:

- đo độ bền của branch/model dưới các hậu kỳ thường gặp ngoài đời.

### 5.4. Drift metrics

Các drift metrics bắt buộc:

- `mean_abs_z_shift`
- `median_abs_z_shift`
- `max_abs_z_shift`

Chúng là **diagnostic**, không phải phán quyết cuối.

Ý nghĩa:

- phát hiện feature nào đổi phân phối mạnh dưới degradation,
- hỗ trợ giải thích calibration risk,
- không được dùng một mình làm gate cứng.

## 6. Nguyên tắc admission

### 6.1. Direct-proxy ban

Feature bị loại ngay nếu phụ thuộc trực tiếp vào:

- `8x8 JPEG lattice`
- explicit chroma bandwidth proxy
- source format / source mode
- missingness cue

### 6.2. Không xét champion ở mức feature đơn

Pha này cấm suy luận kiểu:

- “feature này AUC cao nên đủ để vào champion”

Phán quyết đúng phải ở mức:

- branch
- fusion
- model-level stress test

### 6.3. Conditional cue phải có validity gate

Nếu một cue chỉ sống khi một điều kiện vật lý còn đúng, cue đó:

- không được vào `always-on`,
- phải có `validity score` class-independent,
- phải báo coverage theo split, theo label và theo generator.

## 7. Taxonomy active

Taxonomy active của pha này gồm:

- `always-on`
- `conditional`
- `research-only`
- `drop`

Tổng inventory active để extract hiện tại:

- `35` scalar feature
- `1` validity score

## 8. Nhánh `always-on`

### 8.1. `control_minimal`

Inventory:

1. `frs_mid_variance`
2. `pearson_y_cr`
3. `pearson_y_cb`
4. `pearson_cr_cb`
5. `energy_ratio_chroma`
6. `spatial_snr_ratio`
7. `skew_noise_y`
8. `kurt_noise_y`

Ý nghĩa:

- đây là branch đối chứng tối thiểu,
- utility không quá mạnh,
- nhưng là nhóm ít bẩn nhất trong stack legacy.

Nó đo:

- roughness phổ ở dải giữa
- tương quan màu toàn cục
- chênh lệch residual giữa vùng edge và vùng phẳng
- hình dạng đuôi của phân phối residual `Y`

Branch này rất quan trọng vì:

- nó là baseline sạch hơn để so với mọi family mới.

### 8.2. `fft_midband_y`

Inventory:

1. `fft_mid_logenergy`
2. `fft_mid_flatness`
3. `fft_mid_ring_var`
4. `fft_mid_inner_outer_ratio`
5. `fft_mid_anisotropy_hv`
6. `fft_mid_anisotropy_diag`

Lý do chọn:

- `DCT mid-band` quá codec-direct
- low-frequency quá phụ thuộc scene semantics
- high-frequency quá nhạy với resize/JPEG
- mid-band là vùng cân bằng hợp lý nhất

Family này đo:

- tổng năng lượng ở dải giữa
- mức “spiky vs diffuse” của phổ
- cấu trúc radial trong mid-band
- bất đẳng hướng theo hướng

Đây là family spectral active chính thức của nhánh hiện tại.

## 9. Nhánh `conditional`

### 9.1. `conditional_cfa_rgb`

Inventory:

1. `cfa_rg_pi_xy`
2. `cfa_bg_pi_xy`
3. `cfa_rgb_pi_xy_mean`
4. `cfa_rgb_pi_xy_gap`
5. `cfa_validity_score`

Lý do chọn:

- CFA/demosaicing là cue vật lý mạnh và có ý nghĩa pháp y thật
- nhưng cue này sống gần Nyquist
- nếu ảnh đã bị resize mạnh, evidence đó có thể mất hoàn toàn

Vì vậy family này chỉ được dùng dưới dạng `conditional`.

Tại sao dùng miền `RGB-difference` thay vì `Cr/Cb`:

- `R-G`, `B-G` gần hơn với Bayer difference gốc,
- `Cr/Cb` là tổ hợp tuyến tính và nhạy hơn với chroma processing,
- do đó bản RGB-domain ổn định hơn với codec history.

`cfa_validity_score` có vai trò:

- đo xem bằng chứng high-frequency cần cho CFA còn sống hay không,
- chứ không phải dự đoán nhãn trực tiếp.

## 10. Nhánh `research-only`

### 10.1. `wavelet_decay`

Inventory:

1. `wav_energy_l1`
2. `wav_energy_l2`
3. `wav_energy_l3`
4. `wav_decay_alpha`
5. `wav_ratio_l1_l2`
6. `wav_ratio_l2_l3`

Lý do giữ:

- wavelet vẫn có utility thật,
- nhưng formulation `parent-child correlation` cũ quá resize-fragile.

Family mới chuyển sang đo:

- luật suy giảm năng lượng qua nhiều scale,
- thay vì cố giữ một tương quan tĩnh vốn dễ gãy dưới resize.

### 10.2. `dark_textured_hetero`

Inventory:

1. `lochet_dark_flat_slope`
2. `lochet_dark_flat_r2`
3. `lochet_dark_flat_cv`
4. `lochet_dark_edge_flat_logratio`
5. `lochet_dark_monotone_violation`

Lý do giữ:

- ảnh camera thật thường có signal-dependent noise,
- diffusion images thường đồng đều hơn về residual/noise structure,
- nhưng vùng đo phải được chọn rất cẩn thận trên ảnh web.

Family này hiện chỉ là nhánh nghiên cứu vì utility còn yếu.

### 10.3. `content_adaptive_y_srm`

Inventory:

1. `ysrm_midtex_edge3_energy`
2. `ysrm_midtex_edge3_mar`
3. `ysrm_midtex_square3_energy`
4. `ysrm_midtex_square3_mar`
5. `ysrm_midtex_square5_energy`
6. `ysrm_midtex_square5_mar`

Lý do giữ ở research-only:

- current training baseline cho thấy branch này có utility mạnh,
- nhưng về mặt lý thuyết nó rất dễ hút ringing và block-related nuisance,
- do đó không được đưa thẳng vào champion branch trước khi có audit model-level.

## 11. Các family bị loại

Drop hoàn toàn khỏi active inventory:

- `dct_mid_*`
- chroma SRM
- chroma LBP
- `cross_noise_ratio`
- directional `CFA pi_x / pi_y`

Lý do:

- hoặc là direct proxy,
- hoặc toxic quá rõ so với utility.

## 12. Bảng output của pha này

Bảng đặc trưng active phải chứa:

- các cột định danh:
  - `source_file_path`
  - `patch_path`
  - `generator`
  - `label`
  - `split_role`
  - `dataset_name`
  - `preprocess_version`
  - `feature_version`

- `35` scalar feature
- `1` validity score
- `status`
- `error`

Yêu cầu:

- mọi hàng `status = ok` không được có `NaN` hoặc non-finite
- bảng phải tái lập được theo notebook và API package

## 13. Pha này đảm bảo điều gì

Pha feature hiện hành đảm bảo:

- downstream nhận một feature table có cấu trúc rõ ràng,
- direct proxy đã bị loại khỏi active inventory,
- các cue vi mô được tách ra khỏi cue nền.

Pha này **không đảm bảo**:

- mọi branch hiện có đã đủ sạch để champion-safe,
- clean AUC cao đồng nghĩa deploy-safe,
- model phi tuyến sẽ tự động học đúng signal nếu governance chưa xong.

## 14. Kết luận

Feature extraction hiện hành là một **thiết kế nhiều nhánh có governance**, không phải một bộ feature final đã khóa.

Vai trò đúng của nó là:

- biến các giả thuyết pháp y thành các branch có thể audit,
- cô lập những cue còn đáng nghiên cứu,
- và tạo đầu vào đủ giàu để pha training baseline kiểm tra ở mức hệ thống.
