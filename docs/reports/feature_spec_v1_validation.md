# Validation `old_v1` vs `v4_exact` cho feature spec

> Superseded by [feature_spec_v2_validation.md](C:/Users/USER/Desktop/ai_detector_img/docs/reports/feature_spec_v2_validation.md).

## 1. Mục tiêu

Báo cáo này trả lời 4 câu hỏi:

1. Sau preprocessing v4, các kết luận về feature utility / nuisance có còn đúng khi đo trên cùng tập ảnh đã qua pipeline mới không.
2. Pipeline cũ `old_v1` đã che hay làm méo mức độ shortcut như thế nào.
3. Sau khi loại các feature `DROP`, signal còn lại có đủ lớn để học không.
4. Những family nào cần giữ lại để audit tiếp, những family nào phải bỏ hẳn.

Báo cáo này là bằng chứng thực nghiệm chính để sửa lại
`docs/specs/feature_extraction_standard_v0.md`.

## 2. Protocol

## 2.1. Dữ liệu

Study được chạy trực tiếp trên giao của hai manifest:

- preprocessing cũ: `data/processed/manifest.csv`
- preprocessing mới: `data/processed_v4_rgb248_r4_exact/manifest.csv`

Chỉ giữ lại những `relative_path` được accept bởi cả hai pipeline.
Tổng số dòng chung:

- `85615`

Mẫu dùng cho các task:

- clean LOGO: `1400`
- feature shift: `1120`
- real-only nuisance: `1288`
  - `4:4:4 = 700`
  - `4:2:0 = 588`

Generator được cân bằng trong clean LOGO sample:

- `ADM`
- `GLIDE`
- `Midjourney`
- `SDv14`
- `SDv15`
- `VQDM`
- `Wukong`

Mỗi tổ hợp `generator x label` có `100` ảnh.

## 2.2. Hai pipeline

### `old_v1`

- patch `256 x 256`
- YCrCb patch sau padding / conditional shift / JPEG bottleneck

### `v4_exact`

- patch `248 x 248`
- RGB patch exact crop
- không pad
- không resize
- không JPEG bottleneck

## 2.3. Bộ task

### Clean utility

`label_logo_clean`

- train/test `Leave-One-Generator-Out`
- nhãn `nature` vs `ai`

### Natural nuisance

`real_jpeg_444_vs_420`

- chỉ trên lớp `nature`
- đo khả năng đọc native chroma/JPEG history

### Cross-degradation

Train trên clean, test trên clean image sau khi áp:

- `jpeg95_420`
- `jpeg90_420`
- `resize75_bilinear`
- `resize50_bilinear`
- `resize50_jpeg90_420`

### Feature-shift

Với từng feature `j`, báo cáo:

- `mean_delta`
- `mean_abs_z_shift`
- `median_abs_z_shift`

theo `label x degradation`.

## 2.4. Script và artifact

Script tái lập:

- `script/studies/feature_spec_v1_validation.py`

Artifact:

- `audit_output/studies/feature_spec_v1_validation_20260325/feature_set_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/single_feature_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/feature_shift_metrics.csv`
- `audit_output/studies/feature_spec_v1_validation_20260325/summary.json`

## 3. Kết quả cấp cao

## 3.1. Preprocessing mới làm lộ đúng vấn đề

Top clean AUC theo pipeline:

| Preprocess | Best clean AUC |
|---|---:|
| `old_v1_baseline33` | `0.7195` |
| `old_v1` | `0.7809` |
| `v4_exact` | `0.9236` |

Hai điều không được diễn giải sai:

1. `0.9236` không có nghĩa là `v4_exact` tốt hơn cho champion model.
2. Nó có nghĩa là `v4_exact` bảo toàn được nhiều signal hơn, và đồng thời cũng làm lộ rõ hơn những family đang học shortcut.

Bằng chứng rõ nhất là `crsrm_only`:

- `old_v1`: clean `0.6853`, natural nuisance `0.8824`
- `v4_exact`: clean `0.9236`, natural nuisance `0.9447`
- `v4_exact`: `jpeg90_xdeg = 0.4983`

Tức là family này không "tốt hơn" sau v4.
Nó chỉ được giải phóng khỏi bottleneck cũ, và lộ rõ rằng nó đang shortcut-dominated.

## 3.2. Safe core thật sự tồn tại nhưng không mạnh

`safe_core`:

| Preprocess | Clean | Natural nuisance | JPEG90 xdeg | Resize50 xdeg |
|---|---:|---:|---:|---:|
| `old_v1` | `0.6890` | `0.6641` | `0.6916` | `0.6227` |
| `v4_exact` | `0.6861` | `0.6717` | `0.6850` | `0.6016` |

Đây là kết quả quan trọng nhất của vòng validation này:

- khi bỏ các family toxically codec-coupled,
- signal còn lại vẫn có thật,
- nhưng chỉ ở mức trung bình.

Lo ngại "sau khi loại DROP thì còn quá ít tín hiệu để học" là có cơ sở.

## 3.3. Resize là stress nghiêm trọng hơn extra JPEG cho nhiều forensic feature

Một số feature trên `v4_exact`:

| Feature | Clean | Natural nuisance | JPEG90 xdeg | Resize50 xdeg |
|---|---:|---:|---:|---:|
| `cfa_cr_pi_xy` | `0.7553` | `0.5114` | `0.4914` | `0.5567` |
| `cfa_rg_pi_xy` | `0.7276` | `0.4851` | `0.4998` | `0.5604` |
| `wav_parent_corr_h` | `0.6845` | `0.6548` | `0.6859` | `0.6300` |
| `wav_parent_corr_v` | `0.6864` | `0.6652` | `0.6887` | `0.6119` |
| `ps_alpha` | `0.7021` | `0.6914` | `0.7017` | `0.6683` |

Feature shift xác nhận điều này:

| Feature | `mean_abs_z_shift` JPEG90 nature | `mean_abs_z_shift` Resize50 nature |
|---|---:|---:|
| `cfa_cr_pi_xy` | `0.8451` | `1.4756` |
| `cfa_rg_pi_xy` | `0.7974` | `0.8750` |
| `wav_parent_corr_h` | `0.0428` | `1.8546` |
| `wav_parent_corr_v` | `0.0361` | `1.8848` |
| `ps_alpha` | `0.0113` | `2.4193` |

Hệ quả:

- chỉ đo `JPEG sensitivity` là chưa đủ
- `Cross-Degradation AUC` phải trở thành metric bắt buộc
- nhiều feature pháp y có vẻ "bền vững với JPEG" nhưng lại rất mong manh với resize

## 4. Phán quyết theo family

## 4.1. Chroma microtexture phải loại hẳn

### `crsrm_only`

- `old_v1`: clean `0.6853`, nuisance `0.8824`
- `v4_exact`: clean `0.9236`, nuisance `0.9447`, `jpeg90_xdeg = 0.4983`

### `chroma_lbp_only`

- `old_v1`: clean `0.6565`, nuisance `0.8697`
- `old_v1`: `jpeg90_xdeg = 0.6141`

Kết luận:

- chroma SRM / chroma LBP không còn ở mức "nghi ngờ"
- chúng là shortcut features thật sự

## 4.2. Y-microtexture không được drop blanket

### `ysrm_only`

- `old_v1`: clean `0.7144`, nuisance `0.8088`
- `v4_exact`: clean `0.6995`, nuisance `0.8348`
- `v4_exact`: `jpeg90_xdeg = 0.7084`, `resize50_xdeg = 0.5562`

### `ylbp_only`

- `old_v1`: clean `0.7669`, nuisance `0.8187`
- `v4_exact`: clean `0.7251`, nuisance `0.8609`
- `v4_exact`: `jpeg90_xdeg = 0.7625`, `resize50_xdeg = 0.6946`

Kết luận:

- microtexture trên `Y` có utility thật
- nhưng nuisance vẫn cao
- do đó nó phải vào `priority-audit` hoặc `research-only`, không được auto-drop và cũng không được auto-keep

## 4.3. CFA là family giàu signal nhất, nhưng không safe

Set-level:

- `cfa_chroma_only`
  - `old_v1`: clean `0.5959`
  - `v4_exact`: clean `0.8158`, nuisance `0.8868`
- `cfa_rgb_only`
  - `old_v1`: clean `0.5972`
  - `v4_exact`: clean `0.7865`, nuisance `0.9033`

Single-feature chỉ ra một nuance quan trọng:

- `pi_x / pi_y` directional CFA có nuisance rất cao trên `v4_exact`
- `pi_xy` lại sạch hơn rõ rệt

Ví dụ:

| Feature | Clean | Natural nuisance | JPEG90 xdeg |
|---|---:|---:|---:|
| `cfa_cr_pi_x` | `0.5977` | `0.7834` | `0.5143` |
| `cfa_rg_pi_x` | `0.5987` | `0.7914` | `0.5136` |
| `cfa_cr_pi_xy` | `0.7553` | `0.5114` | `0.4914` |
| `cfa_rg_pi_xy` | `0.7276` | `0.4851` | `0.4998` |

Kết luận:

- không được gọi "CFA family" là một khối đồng nhất
- current directional axes (`pi_x`, `pi_y`) phải loại khỏi champion branch
- `pi_xy` cần được giữ lại để redesign và audit tiếp

## 4.4. Wavelet parent correlation là hướng sạch nhất hiện tại

`wavelet_parent_only`:

| Preprocess | Clean | Natural nuisance | JPEG90 xdeg | Resize50 xdeg |
|---|---:|---:|---:|---:|
| `old_v1` | `0.6863` | `0.6620` | `0.6874` | `0.6247` |
| `v4_exact` | `0.6891` | `0.6646` | `0.6906` | `0.6247` |

Single-feature:

- `wav_parent_corr_h`: clean `0.6845`, nuisance `0.6548`
- `wav_parent_corr_v`: clean `0.6864`, nuisance `0.6652`

Nó không mạnh, nhưng là hướng ổn định và ít gây ngộ nhận nhất hiện tại.

## 4.5. `ps_alpha` cần giữ lại như một feature quarantine

`ps_alpha` trên `v4_exact`:

- clean `0.7021`
- natural nuisance `0.6914`
- `jpeg90_xdeg = 0.7017`
- `resize50_xdeg = 0.6683`

Nó không phản ứng như một feature "extra-JPEG detector" đơn giản.
Nhưng nó vẫn đọc một thành phần native history có thật.

Phán quyết đúng:

- không drop
- không keep
- để `priority-audit / quarantine`

## 4.6. NLF hiện tại thật sự yếu

`nlf_only`:

- `old_v1`: clean `0.5431`
- `v4_exact`: clean `0.4957-0.5628` tùy metric / subset

Kết luận đúng hơn:

- `current 5-feature NLF implementation` là yếu
- điều này chưa đủ để bác bỏ giả thuyết NLF mở rộng

## 5. Bài học cho spec

## 5.1. Champion-safe core phải nhỏ và sạch

Seed set hợp lý nhất hiện tại là:

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

Đây là `safe_core`.
Nó yếu hơn các family mạnh, nhưng sạch hơn và là baseline đúng để phát triển champion.

## 5.2. Muốn có thêm headroom thì phải mở rộng feature space, không thể chỉ "gỡ bớt"

Sau khi loại:

- `dct_mid_*`
- chroma LBP
- chroma SRM
- `cross_noise_ratio`

thì signal sạch còn lại là không đủ để kỳ vọng AUC cao.
Cần thêm family mới, nhưng phải thêm có governance.

## 5.3. Patch aggregation chưa được admit

Báo cáo trước đó đã cho thấy patch aggregation:

- tăng clean AUC
- nhưng đồng thời tăng natural nuisance

Study này cũng xác nhận lại một điểm gần đúng như vậy:

- nhiều feature mong manh với local degradation / resize
- aggregation qua local statistics có nguy cơ khuếch đại shortcut

Patching chỉ nên ở `research branch`, không vào champion notebook hiện tại.

## 6. Backlog feature mới cần nghiên cứu

## 6.1. Ưu tiên audit ngay

- `CFA pi_xy redesign`
- `Y-SRM`
- `expanded NLF`

## 6.2. Ưu tiên nghiên cứu tiếp theo

- chromatic aberration
- PRNU / sensor pattern noise
- resampling periodicity
- patch reliability / patch selection

## 6.3. Cơ sở khoa học

- Bayram, Sencar, Memon: demosaicing artifacts
- Portilla, Simoncelli: wavelet joint statistics
- Thai et al.: generalized noise model
- Johnson, Farid: chromatic aberration
- Lukas, Fridrich, Goljan: PRNU / sensor pattern noise
- Popescu, Farid: resampling traces

Những hướng này đều có cơ sở vật lý hơn so với codec-coupled chroma texture.

## 7. Phán quyết cuối

1. Validation trên cùng tập ảnh đã qua `old_v1` và `v4_exact` xác nhận: preprocessing mới đã làm lộ rõ đúng utility và đúng nuisance.
2. Lo ngại "sau khi bỏ DROP thì còn quá ít signal" là đúng. `safe_core` chỉ đạt clean AUC xấp xỉ `0.686`.
3. `CFA`, `Y-SRM`, `ps_alpha` không được auto-keep, nhưng cũng không được drop blanket.
4. `Wavelet parent correlation` và `safe_core` là baseline sạch nhất hiện tại.
5. Spec feature phải được sửa theo hướng:
   - champion-safe core nhỏ
   - priority-audit branch rõ ràng
   - cross-degradation AUC thành metric bắt buộc
