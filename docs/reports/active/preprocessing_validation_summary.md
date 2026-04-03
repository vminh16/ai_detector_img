# Báo cáo xác thực tiền xử lý v4

## 1. Câu hỏi mà báo cáo này trả lời

Báo cáo này trả lời ba câu hỏi:

1. Tại sao pipeline tiền xử lý cũ là nguồn gốc của shortcut?
2. Tại sao champion path hiện tại phải là `v4_exact`?
3. Sau khi chốt `v4_exact`, những rủi ro nào vẫn còn tồn tại và không được phép che giấu?

Tài liệu này hợp nhất tinh thần và kết luận quan trọng từ:

- [shortcut_risk_validation.md](shortcut_risk_validation.md)
- [preprocessing_pipeline_standard_v4.md](../../specs/archive/preprocessing_pipeline_standard_v4.md)
- các audit trong [audit_output/validation/spec_v4_20260319](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/spec_v4_20260319)

## 2. Bài toán gốc của pha tiền xử lý

Trong dữ liệu web hiện tại:

- ảnh thật chủ yếu là `JPEG`
- ảnh AI chủ yếu là `PNG`

Điều này kéo theo một bất đối xứng cấp dữ liệu:

- ảnh thật thường có lịch sử nén JPEG và subsampling thật,
- ảnh AI thường không có history tương đương.

Nếu đưa dữ liệu đó thẳng vào downstream model, mô hình rất dễ học:

- `JPEG` vs `PNG`
- `4:4:4` vs `4:2:0`
- mode ảnh (`RGBA`, `L`)
- hoặc các artifact do chính pipeline tạo thêm như padding, resize, JPEG round-trip.

Nói cách khác, pha tiền xử lý không chỉ là “làm sạch ảnh”.
Nó là tầng đầu tiên quyết định:

- mô hình có được nhìn thấy bằng chứng thật hay không,
- và pipeline có vô tình bơm thêm shortcut mới hay không.

## 3. Những phát hiện dữ liệu khóa thiết kế

Các audit v4 cho thấy:

- `RGBA` xuất hiện `6,000` ảnh và đều ở fake ADM
- `L` grayscale xuất hiện `781` ảnh và đều ở real
- current 33-feature stack legacy có `label AUC ≈ 0.718` nhưng nuisance real-only `4:4:4 vs 4:2:0 AUC ≈ 0.950`

Điều này dẫn tới ba kết luận rất mạnh:

1. **Mode ảnh là shortcut trực tiếp**, không được phép đi thẳng vào model.
2. **Compression history là nuisance cấp hệ thống**, không thể xem như noise nhỏ.
3. **Preprocessing không được phép tạo artifact mới** chỉ để làm benchmark trông đẹp hơn.

## 4. Vì sao pipeline cũ phải bị loại

Pipeline cũ dùng các thành phần sau:

- padding / resize tùy điều kiện
- crop lệch phase theo logic cũ
- JPEG bottleneck
- một số giả định ngầm về YCrCb là canonical output

### 4.1. Padding và resize

Padding và resize làm sai bài toán theo hai cách:

- tạo pixel mới không tồn tại trong ảnh gốc,
- thay đổi bằng chứng hình học và phổ trước khi feature extraction kịp nhìn thấy dữ liệu.

Về mặt pháp y, đây là hành động nguy hiểm vì:

- nó che mất lỗi thật của feature space,
- đồng thời có thể sinh ra pattern nội suy mới mà model học nhầm.

### 4.2. JPEG bottleneck

Trực giác “nén lại tất cả để công bằng” là sai.

Nếu ký hiệu `H_codec` là history nén/subsampling thật đã in vào pixel domain, thì:

- áp thêm một phép `JPEG(Q)` lên ảnh thật chỉ tạo ra **history chồng lên history cũ**
- áp cùng phép đó lên ảnh AI PNG chỉ tạo ra **history tổng hợp mới**

Hai quá trình này không tương đương.

Vì vậy, `JPEG bottleneck`:

- không xóa được `H_codec`,
- chỉ thay một nuisance bằng nuisance khác,
- và thường còn phá hủy luôn bằng chứng generator/image-formation thật.

### 4.3. Chroma canonicalization

Các audit kiểu `420 down-up` cho thấy:

- `label_logo_auc` giảm rõ
- nhưng nuisance không giảm, thậm chí có trường hợp còn tăng

Nghĩa là canonical hóa chroma kiểu ép tất cả về một chế độ chung:

- không giải quyết gốc vấn đề,
- lại làm mất thêm tín hiệu có ích.

## 5. Cơ sở toán học của `v4_exact`

### 5.1. Nguyên tắc bất biến

Pha tiền xử lý của nhánh này khóa ba bất biến:

1. **Bảo toàn tối đa bằng chứng pixel trên phần patch được giữ lại**
2. **Loại bỏ các shortcut có thể loại bỏ được bằng lý thuyết và kiểm định dữ liệu**
3. **Fail-closed nếu không thể chứng minh an toàn**

Điều này dẫn tới một nguyên tắc rất mạnh:

> Nếu một phép biến đổi làm thay đổi pixel mà không có bằng chứng thực nghiệm rõ ràng rằng nó giảm nuisance nhiều hơn mức nó phá hủy signal, phép đó không được vào champion path.

### 5.2. Tồn tại exact crop

Cho:

- `C`: crop size
- `r`: residue khác `0`
- `A_x(W; C, r) = {x ∈ Z : 0 <= x <= W-C, x ≡ r (mod 8)}`

Khi đó `A_x(W; C, r)` khác rỗng khi và chỉ khi:

`W >= C + r`

Tương tự cho trục `y`.

Ý nghĩa thực dụng:

- không cần padding để “cứu” ảnh nhỏ,
- nếu không tồn tại crop exact đúng residue, ảnh đó phải đi vào `LOW_SUPPORT`,
- fail-closed trung thực hơn rất nhiều so với bơm pixel giả rồi tiếp tục pipeline.

### 5.3. Vì sao chọn `248 @ residue 4`

Trong search space champion v4:

- `C mod 8 = 0`
- `r != 0`
- không padding
- không resize

Audit hiện tại chốt:

- `248` là crop size lớn nhất còn giữ `accepted_ai = 1.0`
- `r = 4` cho khoảng cách pha tới lưới `8x8` lớn nhất

Điều này không có nghĩa `248` là optimum tuyệt đối cho mọi bài toán.
Ý nghĩa đúng là:

- trong search space đang khóa để phục vụ branch hiện tại,
- `248 @ 4` là điểm Pareto tốt nhất giữa coverage, phase safety và khả năng so sánh công bằng downstream.

## 6. Champion preprocessing hiện tại

Pipeline champion path là:

1. đọc byte stream đầu vào
2. decode bằng decoder canonical
3. áp EXIF orientation
4. chuẩn hóa mode:
   - `RGB -> RGB`
   - `RGBA -> RGB` bằng `straight-alpha composite` lên nền `(128,128,128)`
   - `L`, `CMYK`, mode khác -> `UNSUPPORTED_INPUT`
5. tính `S = min(H, W)`
6. nếu `S < 252` -> `LOW_SUPPORT`
7. nếu `S >= 252`:
   - chọn crop exact `248x248`
   - residue `(4,4)`
   - origin là nghiệm hợp lệ gần tâm nhất
8. xuất patch canonical `X_can_rgb8`

Những gì **bị cấm** ở champion path:

- padding
- resize
- JPEG round-trip
- chroma canonicalization
- branch theo format/mode ngoài status fail-closed

## 7. Kết quả thực nghiệm chính

Run preprocessing v4 hiện tại cho:

- `85,615` patch `ACCEPTED`
- `1,573` ảnh `LOW_SUPPORT`
- `783` ảnh `UNSUPPORTED_INPUT`
- `0` `DECODE_ERROR`

Các tỷ lệ quan trọng:

- `P(ACCEPTED) ≈ 0.9819`
- `P(ACCEPTED | ai) = 1.0`
- `P(ACCEPTED | real) ≈ 0.9639`

Điểm quan trọng nhất của các con số này không phải là coverage tuyệt đối cao.
Điểm quan trọng là:

- champion path **không âm thầm sửa dữ liệu** để ép coverage,
- mọi trường hợp không đủ điều kiện đều được tách ra rõ ràng bằng status.

## 8. Những gì v4 đã giải quyết

`v4_exact` đã xử lý được bốn nhóm shortcut lớn:

1. shortcut do mode ảnh
   - `RGBA`
   - `L`

2. shortcut do geometry pipeline
   - padding
   - resize
   - conditional shift

3. shortcut do codec nhân tạo từ pipeline
   - JPEG bottleneck
   - chroma canonicalization

4. shortcut do output contract mơ hồ
   - thống nhất output canonical là `RGB uint8 exact crop`
   - chuyển `YCrCb` thành **derived view** ở pha feature

## 9. Những gì v4 chưa và không thể giải quyết một mình

Đây là phần quan trọng nhất của báo cáo.

`v4_exact` **không thể** tự mình giải quyết:

1. lịch sử nén/subsampling thật đã in sẵn trong ảnh real JPEG
2. ảnh web đã bị resize trước khi vào pipeline
3. khoảng chồng hỗ trợ giữa `real JPEG` và `AI PNG` còn quá ít
4. feature families vốn dĩ rất nhạy với codec history

Do đó, tiền xử lý v4 không được diễn giải như:

- “đã loại xong shortcut”

Nó chỉ được diễn giải là:

- “đã loại bỏ mọi shortcut mà preprocessing có thể loại bỏ một cách trung thực và không phá dữ liệu”.

Phần còn lại phải giao cho:

- feature governance,
- model-level nuisance audit,
- và có thể là thiết kế dữ liệu/slice evaluation tốt hơn.

## 10. Phán quyết

`v4_exact` là champion preprocessing hợp lý nhất hiện nay của nhánh này.

Không phải vì nó tối đa hóa AUC,
mà vì nó đạt được điều quan trọng hơn:

- trung thực với dữ liệu,
- không bơm thêm history nhân tạo,
- và buộc các lỗi thật của feature/model lộ ra sớm.

Đó là lý do pha tiếp theo của nhánh không còn tập trung vào “sửa preprocessing thêm nữa”,
mà chuyển trọng tâm sang:

- tái thiết kế feature space,
- rồi audit model ở mức hệ thống.
