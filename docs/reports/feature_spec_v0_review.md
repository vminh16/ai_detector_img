# Review `feature_extraction_standard_v0`

> Ghi chu: bao cao nay la vong review trung gian. Cac phan quyet cuoi cung da duoc
> cap nhat bo sung trong `docs/reports/feature_spec_v2_validation.md` va da duoc
> phan anh tro lai vao `docs/specs/feature_extraction_standard_v1.md`.

## 1. Mục tiêu

Báo cáo này trả lời bốn câu hỏi còn mở quanh `docs/specs/feature_extraction_standard_v0.md`:

1. Bộ feature trong `spec v0` hiện có quá ít hay chưa.
2. Ảnh hưởng của `JPEG history` lên các feature candidate đã được kiểm chứng đủ chặt chưa.
3. Có nên chia thêm ảnh thành patch ở phase feature extraction không.
4. Nên mở rộng thêm họ feature nào cho vòng thiết kế tiếp theo.

Report này không thay thế `spec v0`. Nó là báo cáo phản biện và mở rộng bằng chứng trước khi chốt `champion feature stack`.

## 2. Nguồn bằng chứng

### 2.1. Artifact trong repo

- `audit_output/studies/feature_spec_v0_review_20260325/feature_set_task_metrics.csv`
- `audit_output/studies/feature_spec_v0_review_20260325/single_feature_task_auc.csv`
- `audit_output/studies/feature_spec_v0_review_20260325/jpeg_sensitivity_shift.csv`
- `audit_output/studies/feature_spec_v0_review_20260325/summary.json`
- `audit_output/validation/spec_v4_20260319/feature_governance_summary.json`
- `audit_output/validation/spec_v4_20260319/keep_only_ablation.csv`
- `audit_output/validation/spec_v4_20260319/feature_governance_family.csv`
- `docs/reports/shortcut_risk_validation.md`
- `docs/reports/feature_space_update.md`

### 2.2. Script tái lập

- `script/studies/feature_spec_v0_review.py`

Script này chạy trực tiếp trên patch canonical sau preprocessing v4:

- input manifest: `data/processed_v4_rgb248_r4_exact/manifest.csv`
- input metadata audit: `audit_output/data_audit/metadata/per_file_metadata.csv`

### 2.3. Paper gốc và tài liệu khoa học

- Bayram, Sencar, Memon, "Classification of digital camera-models based on demosaicing artifacts"  
  https://www.sciencedirect.com/science/article/pii/S1742287608000467

- Portilla, Simoncelli, "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients"  
  https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html

- Thai, Retraint, Cogranne, "Camera model identification based on the generalized noise model in natural images"  
  https://www.sciencedirect.com/science/article/abs/pii/S1051200415003012

- Wang et al., "Double JPEG Compression Detection by Exploring the Correlations in DCT Domain"  
  https://arxiv.org/abs/1806.01571

- Johnson, Farid, "Exposing Digital Forgeries Through Chromatic Aberration"  
  https://farid.berkeley.edu/downloads/publications/acm06c.pdf

- Lukás, Fridrich, Goljan, "Digital Camera Identification From Sensor Pattern Noise"  
  https://ieeexplore.ieee.org/document/1634362

- Popescu, Farid, "Exposing Digital Forgeries by Detecting Traces of Resampling"  
  https://ieeexplore.ieee.org/document/1381775

- Liu et al., "Efficient Source Camera Identification with Diversity-Enhanced Patch Selection and Deep Residual Prediction"  
  https://www.mdpi.com/1424-8220/21/14/4701

## 3. Protocol thực nghiệm mới

## 3.1. Ba bài toán audit

Study mới đo bốn task:

1. `label_logo_clean`
   - phân biệt `nature` vs `ai`
   - LOGO theo `generator_norm`
   - chỉ dùng accepted patch sau preprocessing v4

2. `real_jpeg_444_vs_420`
   - chỉ trong lớp `nature`
   - đo khả năng feature đọc `native chroma subsampling history`

3. `ai_identity_vs_jpeg95_420`
   - chỉ trong lớp `ai`
   - cùng patch nhưng một nhánh giữ nguyên, một nhánh bị nén lại `JPEG q=95, 4:2:0`

4. `ai_identity_vs_jpeg90_420`
   - như trên nhưng `JPEG q=90, 4:2:0`

Hai task cuối không mô phỏng đúng production. Chúng chỉ dùng để đo độ nhạy của feature với một lớp degradation chuẩn hóa, class-independent.

## 3.2. Kích thước mẫu

Từ `summary.json`:

- `label_sample = 720`
  - `120 real + 120 ai` cho mỗi generator trong `midjourney`, `sdv14`, `wukong`
- `natural_nuisance_sample = 1183`
  - `4:4:4 = 600`
  - `4:2:0 = 583`
- `ai_sensitivity_sample = 420`
  - `140` ảnh cho mỗi generator `midjourney`, `sdv14`, `wukong`

Giới hạn quan trọng:

- `sdv15` không xuất hiện trong accepted-ID subset hiện tại của study này.
- Do đó các con số dưới đây là bằng chứng mạnh cho direction design, nhưng chưa phải benchmark cuối cùng cho deploy.

## 3.3. Ba kiểu biểu diễn

Study đo cùng một họ feature dưới ba representation:

- `whole`
  - trích feature trên toàn patch `248 x 248`

- `quad_mean`
  - chia patch thành `2 x 2` subpatch `124 x 124`
  - trích feature trên từng subpatch
  - lấy mean theo feature

- `quad_meanstd`
  - như trên
  - nhưng giữ cả `mean` và `std` theo patch

## 3.4. Các họ feature đã đo

Study này đo:

- `CFA periodicity` 6 feature
- `NLF` 5 feature
- `wavelet` 5 feature
- `keep-global` cũ:
  - `frs_mid_variance`
  - `pearson_*`
  - `energy_ratio_chroma`
  - `spatial_snr_ratio`
  - `skew_noise_*`
  - `kurt_noise_*`
- `controls / quarantine`:
  - `ps_alpha`
  - `ps_deviation_variance`
  - `cross_noise_ratio`

## 4. Kết quả mới: `spec v0` hiện tại chưa đủ rộng, nhưng không được mở rộng mù

## 4.1. Kết luận cấp cao

`spec v0` hiện tại đúng ở chỗ:

- đã loại được các feature toxic nhất của legacy stack,
- đã đặt đúng trọng tâm vào `CFA`, `wavelet`, `NLF`.

Nhưng nó vẫn chưa đủ cho champion cuối vì:

- chưa audit sâu `JPEG history` trên từng family,
- chưa kiểm định hệ quả của patch aggregation,
- còn giữ chung quá nhiều feature khác bản chất trong một nhóm `keep`.

Kết luận đúng không phải là:

- "`spec v0` quá ít nên cứ thêm nhiều feature".

Kết luận đúng là:

- "`spec v0` quá hẹp nếu coi là candidate set cuối",
- nhưng đã đủ để chứng minh rằng nhiều hướng mở rộng tưởng tốt như `CFA` và `patching` vẫn có thể tiếp tục học `JPEG history`.

## 4.2. Bằng chứng định lượng ở mức feature-set

Trên representation `whole`, các set quan trọng có kết quả như sau:

| Feature set | n | Label LOGO AUC | Real 444 vs 420 AUC | AI vs JPEG95 AUC | AI vs JPEG90 AUC |
|---|---:|---:|---:|---:|---:|
| `spec0_keep_plus_controls` | 21 | `0.9024` | `0.8838` | `0.9110` | `0.8564` |
| `cfa_only` | 6 | `0.8886` | `0.8868` | `0.8164` | `0.8080` |
| `spec0_keep_wavelet` | 23 | `0.8844` | `0.8898` | `0.7991` | `0.8119` |
| `spec0_keep` | 18 | `0.8777` | `0.8818` | `0.8019` | `0.8125` |
| `spec0_keep_all_new` | 28 | `0.8771` | `0.8888` | `0.8032` | `0.8166` |
| `spec0_keep_nlf` | 23 | `0.8672` | `0.8801` | `0.8057` | `0.8161` |
| `wavelet_only` | 5 | `0.6578` | `0.6921` | `0.4979` | `0.5177` |
| `nlf_only` | 5 | `0.5628` | `0.6308` | `0.5079` | `0.4678` |

Hai kết luận bắt buộc:

1. `CFA` có utility sạch rất mạnh.
2. Nhưng `CFA` và hầu hết các set chứa nó cũng tách được `JPEG history` gần mạnh ngang tách nhãn.

Nói cách khác:

- `CFA` là feature family quan trọng nhất cần giữ để nghiên cứu tiếp,
- nhưng không còn đủ bằng chứng để auto-promote nó thành `KEEP locked`.

## 4.3. JPEG history đã được chứng minh ảnh hưởng trực tiếp lên nhiều candidate feature

### 4.3.1. Bằng chứng ở mức feature đơn

Một số feature đơn quan trọng:

| Feature | Label AUC | Real 444 vs 420 | AI vs JPEG95 | Mean abs z-shift q95 | Phán quyết sơ bộ |
|---|---:|---:|---:|---:|---|
| `cfa_cr_pi_xy` | `0.7823` | `0.4846` | `0.7653` | `0.4695` | mạnh nhưng JPEG-sensitive |
| `cfa_cr_pi_x` | `0.6925` | `0.7637` | `0.6408` | `0.4631` | rất rủi ro |
| `wav_parent_corr_h` | `0.6659` | `0.6039` | `0.5080` | `0.0259` | ứng viên tốt |
| `wav_parent_corr_v` | `0.6341` | `0.6279` | `0.5070` | `0.0229` | ứng viên tốt |
| `frs_mid_variance` | `0.5557` | `0.5121` | `0.4999` | `0.0023` | an toàn nhưng yếu |
| `ps_alpha` | `0.6840` | `0.6642` | `0.5011` | `0.0063` | natural-history proxy |
| `cross_noise_ratio` | `0.5102` | `0.4374` | `0.7448` | `3.4392` | drop ngay |
| `kurt_noise_cr` | `0.6779` | `0.6875` | `0.6753` | `0.8116` | quarantine/drop |
| `skew_noise_y` | `0.5381` | `0.5317` | `0.4965` | `0.0478` | ứng viên phụ trợ |
| `energy_ratio_chroma` | `0.4742` | `0.4941` | `0.5033` | `0.0067` | rất an toàn nhưng rất yếu |

Điểm cần nhấn mạnh:

- `cross_noise_ratio` không chỉ "hơi nhạy". Nó sụp hoàn toàn dưới synthetic JPEG:
  - `AUC = 0.7448`
  - `mean_abs_z_shift = 3.4392` ở `q95`
  - `mean_abs_z_shift = 8.1564` ở `q90`

- `kurt_noise_cr/cb` và `skew_noise_cr/cb` cũng rất nguy hiểm:
  - label utility không tệ
  - nhưng shift dưới JPEG lớn và nuisance cao

- `frs_mid_variance` và `wav_parent_corr_h/v` là các feature hiếm hoi vừa có utility, vừa ổn định hơn nhiều trước extra JPEG.

### 4.3.2. Giải thích toán học

Nếu viết feature dưới dạng:

`phi(X) = phi_gen(X) + phi_codec(H_c) + eps`

thì feature sẽ nguy hiểm khi:

- `Var[phi_codec(H_c)]` đủ lớn,
- và `H_c` tương quan với nhãn trong dataset.

Task `real_jpeg_444_vs_420` đo trực tiếp khả năng của `phi` trong việc dự báo một thành phần của `H_c`.
Task `ai_identity_vs_jpeg95_420` và `ai_identity_vs_jpeg90_420` đo độ nhạy của `phi` trước một nhiễu codec có kiểm soát.

Khi cả hai cùng cao, ta không còn lý do để gọi feature đó là "image-formation trace" thuần.

### 4.3.3. Một kết quả quan trọng: `ps_alpha` không phản ứng như dự đoán đơn giản

`ps_alpha` có:

- `label AUC = 0.6840`
- `real 444 vs 420 = 0.6642`
- `ai vs JPEG95 = 0.5011`
- `mean_abs_z_shift q95 = 0.0063`

Điều này cho thấy:

- `ps_alpha` không phải là feature "double-JPEG detector" đơn giản,
- nhưng nó vẫn đang đọc một thành phần `native history` thật trong ảnh web.

Do đó:

- không nên `DROP` vội,
- nhưng cũng chưa thể `KEEP`.

Phán quyết đúng là `QUARANTINE`.

## 5. Patch aggregation: có ích cho label, nhưng cũng khuếch đại shortcut

## 5.1. Bằng chứng thực nghiệm

So với `whole`, representation `quad_mean` và `quad_meanstd` làm:

- tăng clean label AUC trên hầu hết set tốt
- nhưng đồng thời tăng mạnh khả năng đọc `real 4:4:4 vs 4:2:0`

Ví dụ:

| Feature set | Whole label | Quad mean label | Quad meanstd label | Whole nuisance | Quad mean nuisance | Quad meanstd nuisance |
|---|---:|---:|---:|---:|---:|---:|
| `spec0_keep` | `0.8777` | `0.8922` | `0.9005` | `0.8818` | `0.9257` | `0.9335` |
| `cfa_only` | `0.8886` | `0.9080` | `0.8981` | `0.8868` | `0.9331` | `0.9332` |
| `spec0_keep_wavelet` | `0.8844` | `0.9012` | `0.9083` | `0.8898` | `0.9258` | `0.9332` |

Với synthetic JPEG, `quad_meanstd` thường còn làm sensitivity tăng thêm:

| Feature set | Whole q95 | Quad meanstd q95 | Whole q90 | Quad meanstd q90 |
|---|---:|---:|---:|---:|
| `spec0_keep` | `0.8019` | `0.8508` | `0.8125` | `0.8760` |
| `cfa_only` | `0.8164` | `0.8350` | `0.8080` | `0.8532` |
| `spec0_keep_wavelet` | `0.7991` | `0.8458` | `0.8119` | `0.8714` |

## 5.2. Giải thích

Gọi `P_k(X)` là subpatch thứ `k`, và `A({phi(P_k)})` là phép tổng hợp patch-level.

Patching không hề đảm bảo:

`I(H_c ; A({phi(P_k)})) <= I(H_c ; phi(X))`

Ngược lại, nếu `H_c` thay đổi cục bộ theo không gian hoặc tương tác với cấu trúc nội dung cục bộ, thì:

- `mean` theo patch thu được local evidence rõ hơn,
- `std` theo patch còn đo trực tiếp spatial heterogeneity của nuisance.

Đó là lý do `quad_meanstd` thường làm:

- label tăng,
- nhưng nuisance cũng tăng.

## 5.3. Phán quyết với patching

Patching không bị loại hoàn toàn.

Nhưng ở thời điểm này:

- không được đưa patch aggregation vào `champion feature extraction`,
- đặc biệt không được đưa `quad_meanstd` vào mặc định.

Nó chỉ nên ở trạng thái:

- `research branch`,
- hoặc `late-stage ablation` sau khi đã khóa xong champion whole-image baseline.

Muốn dùng patching trong champion sau này, cần thêm ít nhất:

1. patch selection dựa trên reliability, không dùng mọi patch như nhau,
2. nuisance-conditioned aggregation, không giữ `std` patch một cách mù quáng,
3. matched-JPEG benchmark riêng cho patch branch.

## 6. Phán quyết mới với từng họ feature

## 6.1. DROP

### `dct_mid_*`

Lý do không đổi:

- direct dependence vào `8 x 8` lattice,
- bản chất là codec-coupled feature.

### `cross_noise_ratio`

Lý do đã được khóa bằng thực nghiệm mới:

- label utility thấp,
- synthetic JPEG sensitivity rất cao,
- z-shift khổng lồ.

Đây là feature phải bỏ ngay khỏi mọi candidate champion set.

### `microtexture chroma` legacy

Bao gồm:

- `LBP Cr/Cb`
- `SRM Cr`
- local chroma texture cũ

Lý do không đổi:

- nuisance rất cao từ các audit trước,
- study mới cho thấy các chroma residual moments cũng tiếp tục hỏng theo đúng hướng này.

## 6.2. QUARANTINE

### `CFA periodicity` theo công thức hiện tại

Đây là thay đổi lớn nhất so với intuition ban đầu.

`CFA` vẫn là family quan trọng nhất về mặt signal sạch.
Nhưng với implementation hiện tại trên `Cr/Cb residual`, nó chưa đủ sạch để auto-keep.

Phán quyết đúng:

- `CFA` là `priority-audit family`,
- không phải `drop`,
- nhưng chưa phải `KEEP locked`.

Đặc biệt:

- `cfa_cr_pi_xy` là feature tốt nhất của họ này,
- `cfa_cr_pi_x`, `cfa_cr_pi_y`, `cfa_cb_pi_x`, `cfa_cb_pi_y` nhạy hơn với native JPEG history.

### `ps_alpha`

Giữ `QUARANTINE`.

Lý do:

- utility có thật,
- synthetic JPEG rất nhỏ,
- nhưng natural history vẫn hiện rõ.

### `ps_deviation_variance`

Giữ `QUARANTINE`.

Lý do:

- utility trung bình,
- natural nuisance dương,
- không nổi bật như `frs_mid_variance`.

### `skew_noise_cr/cb`, `kurt_noise_cr/cb`

Study mới cho thấy nhóm này phải hạ cấp khỏi `keep-global` cũ.

Phán quyết:

- `QUARANTINE`, nghiêng mạnh về `DROP` nếu matched-JPEG audit kế tiếp vẫn xấu.

### `local_color_inconsistency`, `glcm_*_cr`, các local chroma texture khác

Giữ nguyên ở `QUARANTINE`.

Study mới không cứu được trực giác rằng local chroma texture là trục an toàn.

## 6.3. KEEP-CANDIDATE

### `frs_mid_variance`

Đây là feature hiện ra an toàn nhất trong nhóm frequency:

- label utility có thật
- natural nuisance gần chance
- synthetic JPEG gần chance
- z-shift gần như bằng 0

Nó nên ở trong seed set của champion branch.

### `wav_parent_corr_h`, `wav_parent_corr_v`

Đây là kết quả tích cực nhất của vòng review mới.

Hai feature này có:

- label utility tương đối tốt,
- synthetic JPEG sensitivity gần chance,
- z-shift rất nhỏ,
- natural nuisance chỉ ở mức trung bình.

Nếu chỉ chọn một hướng mở rộng ngoài `frs_mid_variance`, hướng đó nên là:

- `wavelet parent-child correlation`,
- không phải cả khối wavelet full.

### Global color correlations

Bao gồm:

- `pearson_y_cr`
- `pearson_y_cb`
- `pearson_cr_cb`
- `energy_ratio_chroma`

Chúng khá yếu nếu đứng một mình, nhưng có lợi ở chỗ:

- synthetic JPEG gần chance,
- shift rất nhỏ,
- ít khả năng học shortcut mạnh kiểu codec lattice.

Chúng phù hợp với vai trò feature phụ trợ trong một set nhỏ, sạch.

### `skew_noise_y`, `kurt_noise_y`, `spatial_snr_ratio`

Study mới cho thấy:

- các moment trên `Y` an toàn hơn rõ rệt so với bản trên `Cr/Cb`,
- `spatial_snr_ratio` có utility không lớn nhưng khá ổn định.

Vì vậy:

- `Y-only residual moments` nên được tách riêng khỏi khối spatial cũ,
- và được giữ ở `KEEP-CANDIDATE`.

## 6.4. RESEARCH-ONLY

### `NLF`

Kết quả mới tiếp tục khẳng định báo cáo cũ:

- `NLF` đúng về mặt vật lý,
- nhưng quá yếu trên ảnh web hiện tại.

Không nên đưa `NLF` vào champion đầu.

### `Chromatic aberration`

Paper của Johnson và Farid cho thấy aberration là một dấu vết optics thật, không phải codec cue.

Tuy nhiên trong repo này có ba rủi ro:

1. preprocessing hiện chỉ giữ một patch trung tâm `248 x 248`, trong khi lateral aberration tăng theo khoảng cách tới optical center,
2. ảnh web thường đã crop/resize, optical center không còn đáng tin,
3. CA cũng có thể bị phá bởi post-processing.

Do đó:

- đáng nghiên cứu,
- nhưng chưa phải feature family ưu tiên số 1.

### `PRNU / sensor pattern noise`

PRNU là dấu vết vật lý rất mạnh trong forensics gốc.
Nhưng với dữ liệu web của repo này:

- patch nhỏ,
- ảnh đã nén/web processed,
- nhiều ảnh AI là PNG render sẵn,
- preprocessing v4 không có nhiều ảnh cùng thiết bị để ước lượng fingerprint.

PRNU vì thế chỉ nên ở trạng thái `research-only`.

### `Resampling / interpolation periodicity`

Đây là feature family có cơ sở từ Popescu-Farid.
Nhưng trong bài toán hiện tại, nó rất nguy hiểm:

- generator pipeline có thể tự tạo resampling-like periodicity,
- web/social pipeline cũng tạo periodicity tương tự,
- feature rất dễ học post-process thay vì generator trace.

Nó phù hợp hơn với:

- robustness benchmark,
- hoặc forgery localization task,
- không phù hợp để đưa thẳng vào champion detector ở giai đoạn này.

## 7. Hướng mở rộng feature space sau vòng review này

## 7.1. Champion seed set nên nhỏ hơn `spec0_keep`

`spec0_keep` hiện vẫn quá rộng, vì nó trộn:

- feature an toàn tương đối,
- feature có utility,
- feature còn nhiễm JPEG history.

Seed set mới nên bắt đầu từ:

1. `frs_mid_variance`
2. `wav_parent_corr_h`
3. `wav_parent_corr_v`
4. `pearson_y_cr`
5. `pearson_y_cb`
6. `energy_ratio_chroma`
7. `skew_noise_y`
8. `kurt_noise_y`
9. `spatial_snr_ratio`

Đây chưa phải champion cuối.
Nó là baseline "sạch hơn" để so sánh với các nhánh mạnh nhưng nguy hiểm hơn.

## 7.2. Nhánh ưu tiên audit tiếp theo

Nhánh cần nghiên cứu tiếp ngay sau seed set là:

1. `CFA periodicity` nhưng viết lại feature cho robust hơn với JPEG
2. `wavelet parent-child` có thể mở thêm hướng `d`
3. `Y-only local co-occurrence / microtexture`

Lý do cho mục 3:

- chroma microtexture hiện độc rõ rệt với codec history,
- trong khi các residual statistic trên `Y` đang cho tín hiệu an toàn hơn.

## 7.3. Thiết kế lại `CFA` thay vì bỏ `CFA`

`CFA` không nên bị drop.
Nhưng implementation hiện tại trên `Cr/Cb residual` chưa đạt champion quality.

Hướng đúng là:

- giữ `CFA` ở trạng thái `priority redesign`,
- thử thêm:
  - `RGB-domain CFA residual`,
  - edge-gated vs flat-gated CFA statistics,
  - matched-JPEG benchmark trước khi admit.

## 8. Các mệnh đề bất biến còn gây shortcut nhưng không thể xóa sạch bằng preprocessing

1. `JPEG / chroma history` là latent variable đã in vào pixel domain.
   - decode không xóa được nó.
   - một transform tất định class-independent sau đó cũng không thể xóa chính xác nó.

2. Ta không nên cố xóa mọi dấu vết image formation.
   - nếu scrub quá mạnh, ta sẽ xóa luôn `CFA`, `sensor-noise`, `wavelet dependency`, và các trace thật cần giữ.

3. Một extra degradation chung không tương đương với history equalization.
   - `real + old history + new JPEG` không giống `fake + new JPEG`.

4. Patching không phải phép "trung hòa".
   - nó khuếch đại cả trace thật lẫn nuisance cục bộ.

5. Ở dữ liệu hiện tại, overlap support giữa `real JPEG` và `ai PNG` vẫn thiếu.
   - do đó feature phase phải được thiết kế như một bài toán `admissibility under nuisance`,
   - không phải chỉ là bài toán "thêm nhiều feature để tăng AUC".

## 9. Phán quyết cuối của vòng review này

1. `spec v0` hiện tại chưa đủ rộng nếu coi là candidate feature stack cuối.
2. Nhưng nó đã đủ để chứng minh rằng:
   - `CFA` cần được giữ,
   - `wavelet parent corr` đáng đẩy lên ưu tiên,
   - `NLF` chỉ nên ở research branch,
   - `patch aggregation` chưa được admit vào champion path,
   - nhiều feature trong `keep-global` cũ phải bị tách lại thành `keep` và `quarantine`.
3. JPEG history đã được kiểm chứng bằng thực nghiệm, không còn là giả thuyết:
   - ở cả natural nuisance task,
   - lẫn synthetic sensitivity task,
   - và ở mức single-feature shift.
4. Bước tiếp theo hợp lý nhất không phải là viết lại toàn bộ extractor một lần.
   - trước hết cần một `feature spec v0.1` mới với seed set nhỏ, sạch hơn,
   - sau đó mới làm notebook/extractor cho nhánh champion và nhánh `CFA redesign` song song.

## 10. Hành động khuyến nghị

1. Viết `feature_extraction_standard_v0.1` với 4 tier:
   - `KEEP-CANDIDATE`
   - `PRIORITY-AUDIT`
   - `RESEARCH-ONLY`
   - `DROP`

2. Champion baseline đầu tiên chỉ dùng:
   - `frs_mid_variance`
   - `wav_parent_corr_h/v`
   - global color correlations
   - `Y-only residual moments`
   - `spatial_snr_ratio`

3. Tách riêng một nhánh audit cho `CFA redesign`.

4. Chưa dùng patch aggregation trong champion notebook.

5. Dùng patching chỉ trong study notebook để trả lời riêng:
   - patch nào đáng giữ,
   - có thể weight patch bằng reliability hay không,
   - và có loại được JPEG-heavy patches trước khi aggregate hay không.
