# Đặc tả tiền xử lý hiện hành

## 1. Trạng thái và phạm vi

Đây là đặc tả active cho pha tiền xử lý của nhánh `codex/preprocessing-v4-core`.

Phiên bản triển khai hiện tại:

- **mã nguồn**: [src/preprocessing](C:/Users/USER/Desktop/ai_detector_img/src/preprocessing)
- **quy ước artifact**: `v4_rgb248_r4_exact`
- **notebook orchestration**: [01_preprocessing.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/01_preprocessing.ipynb)

Tài liệu này tổng hợp và chuẩn hóa nội dung từ:

- [preprocessing_pipeline_standard_v4.md](../archive/preprocessing_pipeline_standard_v4.md)
- [preprocessing_validation_summary.md](../../reports/active/preprocessing_validation_summary.md)

Mục tiêu của tài liệu này là mô tả:

- input contract
- thuật toán canonical
- output contract
- metric và điều kiện đánh giá
- lý do toán học và thống kê đằng sau từng quyết định

## 2. Vai trò của pha tiền xử lý

Pha tiền xử lý không được phép “làm đẹp dữ liệu”.
Vai trò đúng của nó là:

1. loại bỏ các shortcut do **bytes/container/mode** gây ra,
2. tránh tạo thêm artifact nhân tạo từ chính pipeline,
3. chuẩn hóa đầu vào xuống một patch canonical đủ chặt để downstream so sánh được công bằng,
4. fail-closed trong mọi trường hợp chưa chứng minh được an toàn.

Nói ngắn gọn:

> tiền xử lý phải giảm mọi shortcut có thể loại bỏ được một cách trung thực,
> nhưng không được phá hủy thêm bằng chứng pixel thật chỉ để làm benchmark trông đẹp hơn.

## 3. Ký hiệu và định nghĩa

### 3.1. Ký hiệu cơ bản

- `B`: byte stream đầu vào.
- `D(B)`: decoder canonical.
- `X_rgb = D(B)`: ảnh RGB 8-bit sau decode canonical và EXIF orientation.
- `H, W`: chiều cao và chiều rộng của `X_rgb`.
- `S = min(H, W)`.
- `C`: kích thước crop chuẩn.
- `r`: residue theo modulo `8`.
- `A_x(W; C, r) = {x ∈ Z : 0 <= x <= W-C, x ≡ r (mod 8)}`.
- `A_y(H; C, r) = {y ∈ Z : 0 <= y <= H-C, y ≡ r (mod 8)}`.
- `X_can_rgb8`: patch canonical đầu ra của preprocessing.

### 3.2. Trạng thái đầu ra

Mỗi ảnh sau tiền xử lý phải rơi vào đúng một trạng thái:

- `ACCEPTED`
- `LOW_SUPPORT`
- `UNSUPPORTED_INPUT`
- `DECODE_ERROR`

Ý nghĩa:

- `ACCEPTED`: có thể sinh patch canonical đúng chuẩn.
- `LOW_SUPPORT`: ảnh quá nhỏ đối với exact crop đã khóa.
- `UNSUPPORTED_INPUT`: decode được nhưng format/mode chưa được champion path hỗ trợ.
- `DECODE_ERROR`: file hỏng hoặc decoder canonical không mở được.

### 3.3. Derived view

`X_can_rgb8` là output canonical duy nhất của pha này.

Các biểu diễn như:

- `YCrCb`
- residual maps
- FFT
- wavelet

chỉ là **derived views của pha feature extraction**, không phải output của preprocessing.

## 4. Sự kiện dữ liệu khóa thiết kế

Các audit của nhánh này cho thấy:

- ảnh thật chủ yếu là `JPEG`
- ảnh AI chủ yếu là `PNG`
- `RGBA` xuất hiện ở fake ADM nhưng không xuất hiện ở real
- `L` grayscale xuất hiện ở real nhưng không xuất hiện ở fake
- current DSP feature stack cũ đọc mạnh `subsampling / codec history`

Hệ quả:

1. giữ nguyên mode ảnh là shortcut trực tiếp,
2. canonicalization kiểu nén lại/chuyển chroma hàng loạt không giải quyết gốc vấn đề,
3. preprocessing phải tối giản và trung thực.

## 5. Bất biến của preprocessing hiện hành

Ba bất biến active:

1. **Bảo toàn tối đa bằng chứng pixel trên phần patch được giữ lại**
2. **Loại bỏ các shortcut mà preprocessing có thể loại bỏ một cách trung thực**
3. **Fail-closed nếu không chứng minh được an toàn**

Từ đó suy ra bốn nguyên tắc:

- không padding,
- không resize,
- không JPEG bottleneck,
- không chroma canonicalization.

## 6. Input contract

### 6.1. Format được hỗ trợ

Champion path chỉ hỗ trợ:

- `JPEG`
- `PNG`

Mọi format khác:

- nếu decode được nhưng chưa audit -> `UNSUPPORTED_INPUT`
- nếu không decode được -> `DECODE_ERROR`

### 6.2. Mode được hỗ trợ

Champion path chỉ hỗ trợ:

- `RGB`
- `RGBA`

Quy tắc:

- `RGB -> RGB`
- `RGBA -> RGB` bằng **straight-alpha composite** lên nền xám `(128,128,128)`
- `L`, `CMYK` và mode khác -> `UNSUPPORTED_INPUT`

### 6.3. Vì sao phải composite `RGBA`

Nếu giữ nguyên alpha:

- bản thân alpha đã là shortcut nhị phân theo nhãn trên snapshot hiện tại.

Nếu bỏ alpha mà không composite:

- ta mất tính xác định trên pixel nhìn thấy.

Vì vậy, `straight-alpha composite` lên nền trung tính là phương án ít tệ nhất:

- class-independent
- tất định
- không cho alpha đi thẳng vào feature space

## 7. Cơ sở toán học của exact crop

### 7.1. Điều kiện tồn tại crop exact

Cho `C >= 1`, `r ∈ {1,2,...,7}`.

Khi đó:

- `A_x(W; C, r)` khác rỗng khi và chỉ khi `W >= C + r`
- `A_y(H; C, r)` khác rỗng khi và chỉ khi `H >= C + r`

Ý nghĩa thực dụng:

- nếu ảnh không đủ lớn để exact crop đúng residue,
- pipeline không được “cứu” bằng padding hoặc resize,
- mà phải trả về `LOW_SUPPORT`.

### 7.2. Tại sao `r = 4`

Khoảng cách tới lưới `8x8` gần nhất của residue `r` là:

`d(r) = min(r, 8-r)`

`d(r)` đạt cực đại tại `r = 4`.

Nghĩa là:

- nếu buộc phải chọn một residue khác `0`,
- `r = 4` là cách lệch pha khỏi block lattice nhiều nhất.

Điều này là lý do vật lý/phổ quát.
Các tiêu chí như “gần tâm hơn trên snapshot hiện tại” chỉ là tie-break thực nghiệm.

## 8. Quy ước champion path hiện tại

Champion path hiện tại khóa:

- `C = 248`
- `r = 4`
- `support threshold = 252`

Patch canonical được định nghĩa như sau:

1. tìm mọi nghiệm `x` hợp lệ trong `A_x(W; 248, 4)`
2. tìm mọi nghiệm `y` hợp lệ trong `A_y(H; 248, 4)`
3. chọn `x*`, `y*` là nghiệm gần tâm nhất
4. cắt:

`X_can_rgb8 = X_rgb[y*:y*+248, x*:x*+248, :]`

### 8.1. Vì sao chọn `248`

Trong search space champion hiện tại:

- `C mod 8 = 0`
- exact crop
- không resize
- không padding

`248` là crop size lớn nhất vẫn giữ `accepted_ai = 1.0` trên snapshot hiện hành, đồng thời cho label proxy tốt hơn `252` trong các crop audit đã khóa.

Điểm quan trọng:

- đây là optimum trong **search space của champion hiện tại**,
- không phải định lý phổ quát rằng mọi hệ thống đều phải dùng `248`.

## 9. Thuật toán canonical từng bước

Thuật toán active:

1. đọc `B`
2. decode bằng decoder canonical
3. áp EXIF orientation
4. kiểm tra format
5. chuẩn hóa mode theo input contract
6. tính `H, W, S`
7. nếu `S < 252` -> `LOW_SUPPORT`
8. nếu `S >= 252`:
   - tìm `x*`, `y*`
   - exact crop `248x248 @ residue (4,4)`
9. ghi patch `.npy` RGB `uint8`
10. ghi manifest row với đầy đủ metadata vận hành

## 10. Output contract

Với mỗi ảnh, manifest phải ghi tối thiểu:

- đường dẫn nguồn
- đường dẫn patch output
- generator
- label
- status
- input format
- input mode
- mode sau chuẩn hóa
- kích thước gốc
- support
- crop origin
- patch shape/dtype
- preprocess version

Nếu ảnh bị reject:

- vẫn phải có manifest row,
- và mọi stale patch cũ phải bị dọn để tránh contamination.

## 11. Metric bắt buộc của pha tiền xử lý

Pha này phải báo cáo ít nhất:

- số ảnh theo từng `status`
- coverage tổng
- coverage theo `label`
- coverage theo `generator`
- coverage theo `mode`
- `P(ACCEPTED | ai)` và `P(ACCEPTED | real)`
- thống kê `crop_origin_x`, `crop_origin_y`

Điểm quan trọng:

- metric của pha này không chỉ là “bao nhiêu ảnh được giữ lại”,
- mà là “bao nhiêu ảnh được giữ lại mà không đổi dữ liệu theo cách làm sai bài toán”.

## 12. Những phép bị cấm và lý do

### 12.1. Padding

Bị cấm vì:

- tạo pixel mới không tồn tại trong ảnh gốc,
- có thể bơm artifact tại biên,
- che mất failure mode thật của support gate.

### 12.2. Resize

Bị cấm vì:

- thay đổi phổ và cấu trúc residual ngay trong core,
- phá các cue vi mô trước khi feature phase kịp đo,
- tạo interpolation artifact mới.

### 12.3. JPEG bottleneck

Bị cấm vì:

- không xóa được native codec history,
- dễ tạo cue `single- vs double-compressed`,
- phá hủy thêm signal pháp y thật.

### 12.4. Chroma canonicalization

Bị cấm vì:

- trên audit hiện tại làm giảm label utility,
- không làm nuisance biến mất,
- có thể phá thêm bằng chứng liên quan tới image formation.

## 13. Pha tiền xử lý đảm bảo điều gì

Pha này đảm bảo:

- downstream nhận đúng một đầu vào canonical rõ ràng,
- pipeline không tự tạo shortcut lớn mới,
- các case không an toàn bị fail-closed.

Pha này **không đảm bảo**:

- compression history đã biến mất,
- feature space sẽ tự động sạch,
- model cuối sẽ tự động học đúng generator trace.

## 14. Kết luận

Preprocessing hiện hành là một chuẩn **bảo toàn dữ liệu và chống shortcut do pipeline**.

Nó không phải là phép “giải bài toán pháp y” một mình.
Vai trò đúng của nó là:

- dọn sạch mọi thứ mà preprocessing có thể dọn một cách trung thực,
- và bàn giao một patch canonical đủ tin cậy cho pha feature extraction.
