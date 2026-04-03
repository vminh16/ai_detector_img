# Preprocessing Pipeline Standard v4

> Ghi chú trạng thái:
> Đây là bản đặc tả versioned chi tiết của quyết định `v4_exact`.
> Source-of-truth active hiện tại của pha tiền xử lý là
> [preprocessing_standard.md](../active/preprocessing_standard.md).

## 1. Mục tiêu và bất biến

Spec này thay thế bản v3 bằng một contract chặt hơn ở cả hai tầng:

1. preprocessing core,
2. feature governance downstream.

Ba bất biến của v4:

- **Bảo toàn bằng chứng tối đa trên phần patch được giữ lại**: trong core, mọi phép biến đổi làm đổi pixel đều bị cấm mặc định; chỉ cho phép khi audit chứng minh được lợi ích nuisance lớn hơn và chi phí label nhỏ hơn.
- **Loại bỏ mọi shortcut có thể loại bỏ được bằng lý thuyết và kiểm định dữ liệu**: bytes/container, mode, alpha visibility, grayscale visibility, branch theo raw size, padding, resize, conditional shift, JPEG round-trip đều không được đi vào champion path.
- **Fail-closed ở mức hệ thống**: nếu nuisance còn tồn tại ở feature space, preprocessing không được giả vờ là đã giải quyết xong; feature phải vào `KEEP / QUARANTINE / DROP` theo audit.

Audit bundle làm căn cứ cho v4 nằm tại [audit_output/validation/spec_v4_20260319/README.md](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/README.md), chạy trên raw snapshot hash:

`cfd6a842ff6a5e8ad5f78d8321d4bff3700127f993bb262175aca9b93a74e92f`

Snapshot hiện tại có:

- `87,971` file raw trên disk
- `88,007` rows trong `per_file_metadata.parquet`
- `36` rows stale trong metadata cũ

Mọi audit/retrain sau này phải ghi lại snapshot hash tương tự.

## 2. Phạm vi và ký hiệu

### 2.1. Phạm vi

v4 chỉ chuẩn hóa **champion path**. Các đường nghiên cứu phụ có thể tồn tại, nhưng:

- không được ghi đè champion artifact,
- không được dùng để tính metric release,
- không được bypass governance.

### 2.2. Ký hiệu

- `B`: byte stream đầu vào.
- `Y in {0,1}`: nhãn, với `Y=1` là AI, `Y=0` là real.
- `D_v4(B)`: decoder canonical của v4.
- `X_rgb = D_v4(B)`: tensor RGB 8-bit sau decode canonical.
- `H, W`: chiều cao và rộng của `X_rgb`.
- `S = min(H, W)`.
- `C`: crop size.
- `r`: residue khác 0 theo modulo 8.
- `A_x(W; C, r) = {x in Z : 0 <= x <= W-C, x ≡ r (mod 8)}`.
- `A_y(H; C, r) = {y in Z : 0 <= y <= H-C, y ≡ r (mod 8)}`.
- `X_can_rgb8`: patch canonical RGB sau crop exact trên native lattice.
- `T_ycrcb(X_can_rgb8)`: derived view YCrCb dùng cho feature extractor; không phải output core.
- `N_support = 1{S >= C+r}`.
- `N_alpha = 1{input có alpha}`.
- `N_gray = 1{input là grayscale}`.
- `N_sub`: subsampling history thật của real JPEG.

## 3. Sự kiện dữ liệu khóa bởi audit v4

Từ [snapshot_summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/snapshot_summary.json), [geometry_frontier_mult8.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/geometry_frontier_mult8.csv), [feature_governance_family.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/feature_governance_family.csv), [keep_only_ablation.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/keep_only_ablation.csv), [proxy_variant_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/proxy_variant_metrics.csv), [proxy_crop_compare.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/proxy_crop_compare.csv), [proxy_feature_nuisance_auc.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/proxy_feature_nuisance_auc.csv), [input_mode_summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/input_mode_summary.json):

- `248x248 @ residue 4` là crop lớn nhất trong **search space khóa của champion v4** (`C mod 8 = 0`) mà vẫn giữ `accepted_ai = 1.0` trên snapshot hiện tại.
- Với `C=248, r=4`: `P(ACCEPTED)=0.981937`, `P(ACCEPTED|ai)=1.0`, `P(ACCEPTED|real)=0.963869`, `I(Y; N_support)=0.018306 bit`.
- `r=4` là residue tối ưu về **phase distance** trong họ `r in {1,...,7}`. Kết quả `mean_center_linf = 1.368294 px` tốt hơn `r=3` chỉ là empirical tie-break trên snapshot hiện tại, không phải tính chất phổ quát.
- `RGBA` xuất hiện `6000` ảnh và đều ở fake ADM; real không có alpha. Nếu alpha channel hoặc mode được giữ nguyên, đó là shortcut trực tiếp.
- `L` grayscale xuất hiện `781` ảnh và đều ở real; fake không có grayscale. Nếu image mode được giữ nguyên, đó là shortcut trực tiếp.
- Current 33-feature set có `label AUC = 0.717869` nhưng `real-only subsampling nuisance AUC = 0.949724`, nên leakage compression history vẫn là risk cấp hệ thống.
- Sau governance thô hiện tại, `KEEP-only` chỉ còn `label AUC = 0.631846` và `nuisance AUC = 0.701464`, nên feature space admissible hiện tại chưa đủ mạnh để đóng champion model.
- Microtexture family là nguồn nuisance lớn nhất: `delta_lab = 0.020053`, `delta_nuis = 0.203360`, `solo_nuis_auc = 0.944311`.
- Universal chroma canonicalization kiểu `420 down-up` thất bại trên proxy exact-crop audit: `label_logo_auc` giảm `0.811119 -> 0.755013`, trong khi `nuisance_auc` tăng `0.972433 -> 0.975078`.
- `252@4` giữ area tốt hơn nhưng trên proxy crop compare hiện tại cho `label_logo_auc = 0.785574`, thấp hơn `248@4 = 0.811119`; do đó current evidence chưa đủ để mở lại crop-size choice cho champion v4.
- `36` stale metadata rows là vấn đề data hygiene, nhưng không nằm trong current live raw snapshot; mọi audit/train script phải intersect với live raw snapshot trước khi dùng metadata parquet.

## 4. Input Contract

### 4.1. Supported formats

v4 champion core chỉ support:

- `JPEG`
- `PNG`

Lý do:

- toàn bộ corpus audit hiện tại chỉ dùng hai format này,
- mọi format khác chưa có audit symmetry riêng nên chưa được phép đi vào champion path.

Do đó:

- file decode được nhưng format ngoài `JPEG/PNG` -> `UNSUPPORTED_INPUT`
- file corrupt hoặc decoder không mở được -> `DECODE_ERROR`

### 4.2. Supported modes cho champion path

Champion v4 chỉ support:

- `RGB`
- `RGBA`

Các mode sau **không** được vào champion path:

- `L`
- `CMYK`
- mọi mode khác

Lý do:

- `L` gây collapse có cấu trúc trên chroma-derived features và tạo shortcut missingness nếu đưa vào current downstream stack,
- `CMYK` chưa có audit color-management tương thích champion path,
- fail-closed tốt hơn hỗ trợ nửa vời rồi tạo nuisance mới.

Do đó:

- `L` -> `UNSUPPORTED_INPUT` cho champion v4
- `CMYK` -> `UNSUPPORTED_INPUT`

### 4.3. Decoder canonical

Định nghĩa decoder canonical:

`X_rgb = D_v4(B)`

với quy trình chuẩn:

1. Mở file bằng decoder duy nhất của implementation champion.
2. Áp dụng EXIF orientation trước mọi bước hình học.
3. Chuẩn hóa mode:
   - `RGB -> RGB`
   - `RGBA ->` **straight-alpha composite** lên nền xám trung tính `(128,128,128)`, rồi drop alpha
   - `L`, `CMYK`, mode khác -> `UNSUPPORTED_INPUT`

Nền xám `128` được chọn vì:

- tối thiểu hóa bias sáng/tối cực đoan so với nền trắng hoặc đen,
- không để alpha channel đi thẳng vào feature space như một shortcut nhị phân,
- vẫn là một phép tất định duy nhất.

Lưu ý lý thuyết:

- đối với `RGBA`, không tồn tại phép 3-channel nào vừa xóa alpha shortcut vừa bảo toàn tuyệt đối toàn bộ thông tin của cặp `(RGB, A)`;
- do đó compositing là một **contraction bắt buộc**, và vì vậy `N_alpha` phải được xem là nuisance cần audit tiếp ở tầng hệ thống.
- `RGBA` phải được hiểu theo convention **straight alpha** của PNG; implementation không được giả định pre-multiplied alpha.

## 5. Geometry Contract

### 5.1. Định lý tồn tại exact residue crop

Cho `C >= 1`, `r in {1,...,7}`.

Khi đó:

- `A_x(W; C, r)` khác rỗng khi và chỉ khi `W >= C + r`
- `A_y(H; C, r)` khác rỗng khi và chỉ khi `H >= C + r`

**Chứng minh.**

`A_x(W; C, r)` chứa các nghiệm dạng `x = r + 8m` với `x >= 0`. Vì `r > 0` và `x >= 0`, mọi nghiệm hợp lệ đều có `m >= 0`, nên `x >= r`. Nếu tập này khác rỗng thì tồn tại `x <= W-C` và suy ra `W-C >= r`, tức `W >= C+r`.

Chiều ngược lại, nếu `W >= C+r`, chọn ngay `x=r`; khi đó `0 <= r <= W-C` và `x ≡ r (mod 8)`, nên `A_x(W; C, r)` khác rỗng. Tương tự cho trục `y`. QED.

### 5.2. Crop canonical của champion v4

Không gian lựa chọn của **champion v4** bị khóa như sau:

- `C mod 8 = 0`
- `r` phải khác `0`
- không padding
- không resize
- không branch theo size ngoài `ACCEPTED / LOW_SUPPORT`

Quan trọng:

- `C mod 8 = 0` **không phải** định lý hình học của preprocessing core,
- nó là một ràng buộc engineering có chủ đích của champion v4 vì roadmap downstream hiện tại vẫn cần một patch size tương thích với các extractor block-based và các audit so sánh ngang hàng,
- các ứng viên `C` không chia hết cho `8` như `252` không bị chứng minh là sai về mặt preprocessing; chúng chỉ nằm ngoài search space khóa của champion v4.

Cho ảnh `X_rgb` có kích thước `(H, W)`, định nghĩa:

- `C* = 248`
- `r* = 4`

Patch canonical tồn tại khi và chỉ khi:

`S = min(H,W) >= 252`

Khi đó crop origin được chọn là nghiệm gần tâm nhất:

- `x* = argmin_{x in A_x(W; 248, 4)} |x - (W-248)/2|`
- `y* = argmin_{y in A_y(H; 248, 4)} |y - (H-248)/2|`

và:

`X_can_rgb8 = X_rgb[y*:y*+248, x*:x*+248, :]`

### 5.3. Vì sao chọn `248 @ residue 4`

#### Mệnh đề 1

Trong search space khóa của champion v4, tức họ `C mod 8 = 0`, crop lớn nhất có `P(ACCEPTED | ai) = 1` trên raw snapshot hiện tại là `C = 248`.

**Chứng cứ dữ liệu.**

Từ [geometry_frontier_mult8.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/geometry_frontier_mult8.csv):

- `256` thất bại với `accepted_ai = 0.590835`
- `248` đạt `accepted_ai = 1.0`
- mọi `C < 248` đều giữ ít evidence hơn mà không cần thiết

Do đó, trong search space khóa của champion v4, `248` là lựa chọn evidence-preserving tốt nhất vẫn tránh được low-support ở class AI.

#### Ghi chú ngoài search space

Ứng viên `252` có area tốt hơn về mặt hình học, nhưng current proxy crop compare tại [proxy_crop_compare.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/proxy_crop_compare.csv) cho thấy:

- `248@4`: `label_logo_auc = 0.811119`
- `252@4`: `label_logo_auc = 0.785574`

Vì vậy v4 không thể kết luận `252` tốt hơn chỉ từ area ratio. Câu hỏi `248` hay `252` chỉ nên được mở lại sau khi có feature set admissible mới.

#### Mệnh đề 2

Với lattice `8x8`, độ lệch pha khỏi biên `0 mod 8` của residue `r` là:

`d(r) = min(r, 8-r)`

`d(r)` đạt cực đại duy nhất tại `r = 4`.

Hệ quả:

- `r=4` là residue xa biên `0 mod 8` nhất có thể,
- nếu mục tiêu là triệt tiêu phụ thuộc vào grid phase mà vẫn giữ crop exact trên native lattice, `r=4` là lựa chọn chuẩn.

#### Mệnh đề 3

Trong các residue cho `C=248`, `r=4` đồng thời có center-drift thực nghiệm tốt nhất **trên snapshot hiện tại**.

Từ [geometry_residue_scan_crop_248.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/geometry_residue_scan_crop_248.csv):

- `r=3`: `mean_center_linf = 2.046338`
- `r=4`: `mean_center_linf = 1.368294`

Vì vậy `r=4` không chỉ đúng về lý thuyết phase distance mà còn tốt hơn về giữ crop gần tâm trên corpus hiện tại. Kết luận này là empirical tie-break, không phải property phổ quát của mọi distribution kích thước ảnh.

## 6. Output Contract

Status của v4 chỉ có bốn giá trị:

- `ACCEPTED`
- `LOW_SUPPORT`
- `UNSUPPORTED_INPUT`
- `DECODE_ERROR`

### 6.1. Nếu `status = ACCEPTED`

Output core bắt buộc gồm:

- `x_can_rgb8`: patch `248x248x3`, dtype `uint8`
- `crop_size = 248`
- `residue = (4,4)`
- `crop_origin = (x*, y*)`
- `support_threshold = 252`
- `input_mode in {RGB, RGBA}`
- `alpha_policy = straight_alpha_over_gray128` nếu input là `RGBA`, ngược lại `none`
- `preprocess_version = v4`

### 6.2. Nếu `status = LOW_SUPPORT`

Output bắt buộc gồm:

- `x_can_rgb8 = null`
- `reason = low_support`
- `support_threshold = 252`

Không tồn tại patch canonical cho `LOW_SUPPORT`, nên:

- champion classifier không được chạy,
- mọi metric patch-level trên nhóm này là không hợp lệ,
- audit cho nhóm này chỉ được làm ở mức system bit `N_support`, coverage, abstention và end-to-end operating point.

## 7. Các phép bị cấm trong Champion Core

Các phép sau bị cấm tuyệt đối trong champion core:

- `reflect / replicate / constant padding`
- mọi `resize`, kể cả resize short-side rồi crop
- mọi `conditional +k` phụ thuộc raw coordinate
- mọi `JPEG bottleneck` hoặc `round-trip`
- mọi `chroma420 canonicalization`
- mọi branch classifier theo raw size, format, image mode, alpha flag

Lý do:

- padding, resize, conditional shift, JPEG round-trip đã được chứng minh hoặc kiểm định là tạo confound hoặc information loss không được phép,
- chroma canonicalization kiểu universal `420 down-up` đã thất bại trên exact-crop proxy audit hiện tại.

## 8. Derived Views và nguyên tắc bảo toàn thông tin

Core của v4 chỉ chuẩn hóa `X_can_rgb8`.

Mọi view khác như:

- `T_ycrcb(X_can_rgb8)`
- wavelet view
- CFA residual view
- residual maps

đều thuộc **feature phase**, không thuộc preprocessing core.

Điều này sửa một lỗi thiết kế của v3:

- nếu core lưu thẳng `YCrCb`, nó tự áp đặt một contraction và một rounding path không cần thiết,
- trong khi invariant của preprocessing là giữ bằng chứng pixel càng nguyên vẹn càng tốt.

Do đó:

- artifact canonical của v4 là RGB patch,
- feature extractor nào cần `YCrCb` phải tự sinh view này từ `X_can_rgb8` bằng một transform tất định đã được khóa trong spec feature tương ứng.

## 9. System Metrics bắt buộc

Mọi benchmark release phải report song song:

- `coverage = P(status = ACCEPTED)`
- `coverage_y = P(status = ACCEPTED | Y=y)` cho cả `y=0,1`
- `I(Y; N_support)`
- metric conditional trên tập `ACCEPTED`
- metric end-to-end ở mức operating point

Với ngưỡng quyết định `tau`, định nghĩa:

- `system_hit_rate(tau) = P(score >= tau, status = ACCEPTED | Y=1)`
- `system_false_alarm_rate(tau) = P(score >= tau, status = ACCEPTED | Y=0)`

Hai đại lượng này:

- **không phải** standard TPR/FPR của ROC literature,
- không được gọi là ROC coordinates,
- không được dùng để báo `AUC` so sánh trực tiếp với paper bên ngoài.

Do đó mọi report hợp lệ phải tách rõ:

- ROC/AUC conditional trên tập `ACCEPTED`
- coverage và support bias
- `system_hit_rate/system_false_alarm_rate` ở operating point triển khai

Chỉ report metric trên `ACCEPTED` mà không report coverage và system operating characteristics là không hợp lệ.

## 10. Feature Governance

### 10.1. Đơn vị governance

Đơn vị chính của governance là **feature family**, không phải feature đơn lẻ.

Lý do:

- leakage thường phân tán trên nhiều feature tương quan,
- leave-one-feature-out thường đánh giá thấp nuisance contribution thật của cả cụm,
- audit v4 hiện tại xác nhận điều này: microtexture family làm giảm nuisance mạnh ở mức family (`0.203360`) trong khi nhiều feature đơn lẻ chỉ có `delta_nuis` rất nhỏ.

### 10.2. Benchmark governance

Hai benchmark bắt buộc:

- `B_lab^v4`: benchmark label sạch, chỉ dùng `ACCEPTED`, metric chính là `AUC_LOGO` theo generator
- `B_sub^v4`: benchmark nuisance real-only `4:4:4` vs `4:2:0`

Với tập feature hiện hành `S` và family/feature `g`, định nghĩa:

- `Delta_lab(g) = A_lab(S) - A_lab(S \ g)`
- `Delta_nuis(g) = A_nuis(S) - A_nuis(S \ g)`

CI phải được ước lượng bằng paired bootstrap trên cùng test split hoặc cùng out-of-fold prediction set.

### 10.3. Ba tier

#### `DROP`

`g` vào `DROP` nếu thỏa ít nhất một trong các điều kiện:

1. `g` nhìn trực tiếp vào biến bị cấm về mặt lý thuyết:
   - bytes/container/metadata
   - raw size / support bit như một branch classifier
   - image mode / alpha flag như feature trực tiếp
2. `CI_low(Delta_nuis(g)) > 0` và `CI_high(Delta_lab(g)) <= 0`
3. `g` phụ thuộc trực tiếp vào một nuisance lattice/codec đã biết mà không có audit clean nào chứng minh được tính admissible

Định nghĩa “phụ thuộc trực tiếp”:

- công thức feature explicit partition dữ liệu theo primitive nuisance,
- hoặc explicit đo năng lượng/statistic trên primitive đó.

Ví dụ:

- `dct_mid_*` là direct vì dùng block DCT `8x8` cố định,
- `frs_mid_variance` không phải direct theo nghĩa này vì nó dùng Fourier rings toàn patch, dù vẫn có thể bị ảnh hưởng gián tiếp bởi codec.

Trong repo hiện tại, `dct_mid_*` phải được xem là `DROP` cho champion candidate vì phụ thuộc trực tiếp vào block grid `8x8` và không có audit clean đủ mạnh để admissible.

#### `QUARANTINE`

`g` vào `QUARANTINE` nếu:

1. `CI_low(Delta_nuis(g)) > 0`, dù `Delta_lab(g)` vẫn dương
2. hoặc family chứa `g` có `CI_low(Delta_nuis(family)) > 0`
3. hoặc có coupling lý thuyết rõ với compression history / chroma bandwidth / residual codec

v4 không dùng ngưỡng cứng kiểu `A_nuis^solo >= 0.65` làm decision rule, vì ngưỡng đó không có justification đủ mạnh và phụ thuộc benchmark.

`QUARANTINE` nghĩa là:

- được giữ cho nghiên cứu,
- được log trong ablation,
- không được đưa vào champion model.

#### `KEEP`

`g` chỉ vào `KEEP` khi đồng thời:

1. không nằm trong `DROP`
2. không nằm trong `QUARANTINE`
3. đã được re-audit trên benchmark v4 sạch sau khi re-extract feature từ `X_can_rgb8`

### 10.4. Phán quyết cho feature space hiện tại

Từ [feature_governance_family.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/feature_governance_family.csv), [feature_governance_feature_ci.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/feature_governance_feature_ci.csv), [proxy_feature_nuisance_auc.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/proxy_feature_nuisance_auc.csv):

- `microtexture` -> `QUARANTINE`
  - `Delta_lab = 0.020053`
  - `Delta_nuis = 0.203360`
  - `solo_nuis_auc = 0.944311`
- `ps_deviation_variance` -> `QUARANTINE`
  - `Delta_nuis CI = [0.000904, 0.010547]`
- `cross_noise_ratio` -> `QUARANTINE`
  - `Delta_nuis CI = [0.000435, 0.002510]`
- `ps_alpha` -> `QUARANTINE`
  - hiện tại nó có label utility, nhưng trên exact-crop proxy audit vẫn cho `nuisance_auc_abs = 0.706122`
- `dct_mid_*` -> `DROP`
- `color family` và các spatial moments còn lại mới chỉ là ứng viên `KEEP`, chưa phải `KEEP`; chúng phải được re-extract và re-audit dưới v4 trước khi retrain champion

Ablation bắt buộc tại [keep_only_ablation.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319/keep_only_ablation.csv) cho thấy:

- `full33`: `label AUC = 0.717869`, `nuisance AUC = 0.949724`
- `KEEP-only` hiện tại: `label AUC = 0.631846`, `nuisance AUC = 0.701464`

Hệ quả:

- governance direction là đúng,
- nhưng feature pool admissible hiện tại chưa đủ để train champion model có ích.

Hệ quả trực tiếp:

- champion artifact v1 hiện tại không còn admissible dưới v4,
- bắt buộc phải re-extract features, re-audit và retrain,
- và trước khi retrain phải có thêm feature families mới hoặc redesign feature stack.

## 11. Quy tắc admit cho mọi universal suppressor

Cho một operator pixel-changing `G`.

`G` chỉ được vào champion core nếu đồng thời:

1. `G` là operator **universal**, không branch theo dữ liệu
2. `G` giảm mọi nuisance benchmark mục tiêu với bằng chứng thống kê:
   - `A_nuis_k(G(X)) < A_nuis_k(X)` cho mọi nuisance chính `k`
3. `G` không làm giảm label benchmark vượt budget:
   - `A_lab(G(X)) >= A_lab(X) - delta_lab_max`
4. `G` không vi phạm invariant bảo toàn bằng chứng ở mức lớn hơn lợi ích nuisance đã chứng minh

Trên audit hiện tại, operator `chroma420` bị loại vì:

- `label_logo_auc`: `0.811119 -> 0.755013`
- `nuisance_auc`: `0.972433 -> 0.975078`

Do đó v4 **không admit bất kỳ universal suppressor pixel-changing nào** ngoài canonical alpha composite bắt buộc cho `RGBA`.

## 12. Định nghĩa Proxy Audit

Trong bundle v4, “proxy raw audit” nghĩa là:

- exact-crop trực tiếp trên raw snapshot,
- sample cân bằng theo generator/label,
- chỉ dùng một proxy feature set nhỏ,
- metric label chính là `LOGO AUC`.

Nó có mục đích:

- kiểm tra hướng tác động của quyết định preprocessing,
- không thay thế benchmark chính của full feature stack,
- không được nhầm với `train_core -> id_test` AUC của current 33-feature dataset.

## 13. Pipeline Cuối Cùng

Pipeline tối ưu cuối cùng của v4, trong phạm vi thiết kế đã audit, là:

1. Nhận `B`.
2. Nếu format không thuộc `JPEG/PNG` -> `UNSUPPORTED_INPUT`.
3. Decode bằng canonical decoder duy nhất của champion implementation.
4. Áp dụng EXIF orientation.
5. Chuẩn hóa mode:
   - `RGB -> RGB`
   - `RGBA -> RGB` bằng **straight-alpha composite** lên nền `(128,128,128)`
   - mode khác -> `UNSUPPORTED_INPUT`
6. Tính `H, W, S = min(H,W)`.
7. Nếu `S < 252` -> `LOW_SUPPORT`, không sinh patch, champion model không chạy.
8. Nếu `S >= 252`:
   - crop exact native-lattice với `C=248`, residue `(4,4)`
   - chọn origin gần tâm nhất
   - xuất `X_can_rgb8`
9. Mọi feature extractor downstream phải đọc từ `X_can_rgb8`.
10. Chỉ feature/family ở trạng thái `KEEP` mới được đi vào champion model.
11. `L` grayscale hiện tại không đi vào champion v4; nếu cần hỗ trợ, phải có grayscale-safe feature branch và benchmark riêng.

Trên snapshot hiện tại, policy champion này cho:

- `P(champion_ready | ai) = 1.0`
- `P(champion_ready | real) = 0.946429`
- `P(champion_ready) = 0.973218`

### 13.1. Những gì không còn được phép

- không `reflect pad`
- không `resize`
- không `conditional +3`
- không `JPEG bottleneck`
- không `native_chroma420`
- không metrics chỉ trên `ACCEPTED` mà thiếu coverage/system operating characteristics
- không champion model dùng nguyên bộ 33 feature hiện tại

### 13.2. Phiên bản khóa

Tên logic đề xuất cho artifact:

`preprocess_version = v4_rgb248_r4_exact`

## 14. Kết luận

v4 không còn là một spec chỉ “trung thực về những gì preprocessing không làm được”. Nó là một contract đầy đủ hơn:

- loại bỏ các shortcut có thể loại bỏ ở core,
- khóa input contract để mode/alpha/grayscale không rò trực tiếp vào model,
- chọn geometry tối ưu bằng tiêu chí có thể kiểm định,
- buộc feature space đi qua governance ba tầng,
- và fail-closed ở mức hệ thống nếu nuisance còn sống ở downstream.

Trong trạng thái hiện tại của repo, bước tiếp theo đúng là:

1. implement preprocessing core đúng theo v4,
2. re-extract feature trên `X_can_rgb8`,
3. audit lại feature families theo rule `KEEP / QUARANTINE / DROP`,
4. retrain champion model mới.
