# Báo cáo xác thực không gian đặc trưng

## 1. Mục tiêu của báo cáo

Báo cáo này trả lời các câu hỏi:

1. Vì sao stack 33 đặc trưng DSP cũ không còn đủ tin cậy?
2. Vì sao kết quả `0/0 pass gate` không có nghĩa là “không còn gì để học”?
3. Tại sao spec hiện tại chọn kiến trúc `always-on + conditional + research-only`?

Tài liệu này hợp nhất nội dung quan trọng từ:

- [feature_space_update.md](feature_space_update.md)
- [feature_spec_v0_review.md](feature_spec_v0_review.md)
- [feature_spec_v1_validation.md](feature_spec_v1_validation.md)
- [feature_spec_v2_validation.md](feature_spec_v2_validation.md)
- [feature_extraction_standard_v2.md](../../specs/archive/feature_extraction_standard_v2.md)

## 2. Vấn đề của baseline 33-feature cũ

Baseline cũ được xây trên một trực giác pháp y hợp lý:

- phổ tần số,
- tương quan màu,
- residual microtexture,
- normalized spatial statistics.

Vấn đề không nằm ở chỗ các ý tưởng này “sai hoàn toàn”.
Vấn đề nằm ở chỗ, trên dữ liệu web hiện tại:

- rất nhiều feature đọc `H_codec` mạnh hơn `H_gen`,
- và một số feature gần như trở thành proxy trực tiếp cho subsampling/JPEG history.

Biểu hiện thực nghiệm:

- label AUC của stack cũ không tệ,
- nhưng nuisance AUC còn cao hơn nhiều.

Điều đó có nghĩa:

- representation vẫn có signal,
- nhưng signal sạch và signal bẩn đang bị trộn vào nhau.

## 3. Tại sao `0/0 pass gate` không phải kết luận cuối

Một vòng validation trước đây cho kết quả:

- `0` feature đơn pass strict gate
- `0` feature-set pass champion gate

Đây là một kết quả quan trọng, nhưng phải hiểu đúng:

- nó bác bỏ framing “tìm một feature đơn hoàn hảo”
- nó không bác bỏ ý tưởng dùng **tập đặc trưng nhiều nhánh**
- nó không bác bỏ downstream fusion phi tuyến có governance

Ba lý do:

1. nhiều cue pháp y là **conditional**
   - ví dụ `CFA` chỉ có ý nghĩa khi evidence Nyquist còn sống

2. nhiều cue yếu nhưng bổ sung nhau
   - từng feature đơn lẻ không vượt ngưỡng cao
   - nhưng khi ghép branch hợp lý, utility tăng rõ

3. “không bị nén ảnh tác động” là một yêu cầu vật lý quá mạnh
   - nhất là với các cue vi mô như CFA, SRM, wavelet cấp cao

## 4. Phân tích theo họ đặc trưng

### 4.1. `control_minimal`

Nhóm này gồm:

- `frs_mid_variance`
- `pearson_y_cr`
- `pearson_y_cb`
- `pearson_cr_cb`
- `energy_ratio_chroma`
- `spatial_snr_ratio`
- `skew_noise_y`
- `kurt_noise_y`

Ý nghĩa:

- đây là nhóm ít bẩn nhất trong stack legacy,
- nhưng utility đơn độc của nó khá yếu,
- vì vậy nó chỉ nên đóng vai trò **lower-bound control branch**.

Nó quan trọng vì:

- cho ta một mốc “nếu chỉ dùng tín hiệu tương đối sạch thì bài toán còn khó đến mức nào”,
- từ đó đánh giá family mới có thật sự thêm signal sạch hay chỉ thêm shortcut.

### 4.2. `fft_midband_y`

Đây là family spectral mới được đưa vào active spec.

Lý do chọn:

- `DCT mid-band` dính trực tiếp vào lưới `8x8` JPEG
- low-frequency bị scene semantics chi phối
- high-frequency/Nyquist bị resize và JPEG phá mạnh nhất
- mid-band là vùng thỏa hiệp tốt nhất để đo sự tổ chức năng lượng phổ mà ít codec-direct hơn

Family này đo:

- năng lượng mid-band
- độ phẳng/phân tán của phổ
- độ gồ ghề radial trong dải giữa
- bất đẳng hướng theo hướng

Điểm cần nhớ:

- `fft_midband_y` không “miễn nhiễm JPEG”
- nó chỉ **ít codec-direct hơn** nhiều family cũ

### 4.3. `conditional_cfa_rgb`

Đây là family mạnh nhất về mặt vật lý nhưng cũng mong manh nhất.

`CFA` đo dấu vết periodicity `2x2` của quá trình demosaicing/Bayer.
Trên ảnh chưa bị resize nặng, đây là một cue thật.

Tuy nhiên:

- CFA sống gần Nyquist,
- resize low-pass có thể xóa hẳn evidence này.

Do đó, cách dùng đúng không phải:

- “làm CFA robust với resize bằng mọi giá”

Mà là:

- chuyển CFA thành **conditional branch**,
- chỉ bật khi `cfa_validity_score` cho thấy bằng chứng high-frequency vẫn còn.

### 4.4. `wavelet_decay`

Current wavelet formulation không còn dùng `parent-child correlation` tĩnh như trước.

Lý do:

- correlation giữa hai scale kề nhau gãy mạnh khi ảnh bị resize,
- nhưng quy luật suy giảm năng lượng qua nhiều scale vẫn còn là một đối tượng hợp lý để đo.

Family `wavelet_decay` vì vậy chuyển sang đo:

- năng lượng ở level 1, 2, 3
- slope suy giảm qua scale
- tỷ lệ năng lượng giữa các level

Family này vẫn ở `research-only` vì:

- có utility thật,
- nhưng độ bền dưới resize vẫn cần audit thêm ở mức model.

### 4.5. `dark_textured_hetero`

Ý tưởng của family này là:

- ảnh thật có nhiễu phụ thuộc tín hiệu,
- ảnh diffusion thường có residual/noise structure đồng đều hơn.

Điểm khó là:

- web JPEG phá rất mạnh vùng phẳng và vùng cạnh sắc,
- nên nếu đo dị phương sai trên toàn ảnh, signal thật dễ bị chìm.

Nhánh hiện tại chỉ đo trên:

- vùng tối
- có texture vừa phải
- không quá phẳng
- không phải cạnh sắc cực mạnh

Kết quả hiện tại chưa mạnh, nên family này vẫn ở `research-only`.

### 4.6. `content_adaptive_y_srm`

Đây là nơi có mâu thuẫn lớn nhất.

Một mặt:

- `Y-SRM` có utility rất mạnh ở training baseline hiện tại.

Mặt khác:

- về lý thuyết nó rất dễ hút ringing, block-boundary residue và codec artifacts,
- và audit feature-level trước đó cho thấy các residual family có thể rất bẩn.

Chiến lược hiện tại không phải là drop hẳn.
Chiến lược đúng là:

- giữ `content-adaptive Y-SRM` ở `research-only`,
- nhưng bắt buộc audit thêm ở mức model và ablation.

## 5. Các family bị loại

Những family bị loại khỏi active spec không phải vì “trông không đẹp”.
Chúng bị loại vì hoặc:

- là direct proxy,
- hoặc toxic quá rõ so với lợi ích.

Bao gồm:

- `dct_mid_*`
- chroma SRM
- chroma LBP
- `cross_noise_ratio`
- directional `CFA pi_x / pi_y`

Điểm chung của nhóm này là:

- hoặc dính rất trực tiếp vào `8x8 lattice`,
- hoặc phản ứng quá mạnh với chroma bandwidth/subsampling,
- hoặc cho clean AUC cao nhưng nuisance gần như cao tương đương.

## 6. Vì sao spec hiện tại chọn kiến trúc nhiều nhánh

Nếu ép mọi feature phải là:

- always-on
- codec-agnostic
- resize-robust
- strong clean utility

thì gần như không family nào sống sót.

Điều đó cho thấy lỗi nằm ở framing, không chỉ ở feature.

Spec hiện tại chọn kiến trúc nhiều nhánh vì:

1. `always-on` branch giữ baseline sạch hơn
2. `conditional` branch cho phép dùng cue vi mô khi còn hợp lệ
3. `research-only` branch giữ không gian tìm kiếm mở nhưng không được đi thẳng vào champion path

Đây là cách duy nhất để vừa trung thực với vật lý ảnh,
vừa không tự đánh đồng mọi cue vi mô là “vô dụng”.

## 7. Những gì feature phase đảm bảo và không đảm bảo

Feature phase hiện tại đảm bảo:

- input contract sạch và nhất quán từ preprocessing v4
- feature families được đặt trong taxonomy rõ ràng
- direct proxies đã bị loại khỏi active inventory

Nhưng feature phase **không đảm bảo**:

- model cuối sẽ tự động học generator trace thay vì nuisance,
- mọi branch active đều đã đủ sạch để release,
- clean AUC cao đồng nghĩa deploy-safe.

Đó là lý do phải có thêm:

- training baseline benchmark,
- model-level nuisance audit,
- và branch ablation.

## 8. Kết luận

Kết luận quan trọng nhất của pha feature không phải là:

- “đã tìm ra bộ đặc trưng cuối cùng”

Mà là:

- “đã xây được một không gian đặc trưng có cấu trúc, đủ sạch để audit ở mức hệ thống”.

Nói cách khác:

- preprocessing v4 buộc các lỗi thật của feature lộ ra,
- feature v2 biến các giả thuyết đó thành branch cụ thể,
- và giờ chỉ còn một bước quan trọng nữa:
  kiểm tra xem branch nào thực sự sống sót sau audit ở mức model.
