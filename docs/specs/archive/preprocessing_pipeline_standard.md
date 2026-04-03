# Đặc tả v3: Pipeline Tiền Xử Lý Chuẩn và Quy Trình Audit Nuisance

## 1. Phạm vi

Tài liệu này chuẩn hóa tầng tiền xử lý cho detector hiện tại với ba mục tiêu chính:

1. bảo toàn tối đa bằng chứng pixel có thật,
2. loại bỏ các shortcut do chính pipeline tạo ra,
3. tách bạch phần có thể chứng minh bằng toán học khỏi phần chỉ mới được hỗ trợ bởi thực nghiệm.

Tài liệu này **không** khẳng định những điều sau:

- không có phép tiền xử lý tất định nào có thể xóa chính xác mọi lịch sử nén JPEG chưa biết;
- không có cơ sở để nói rằng một họ biến đổi "class-blind" theo nghĩa hẹp sẽ làm phân phối đầu ra độc lập với nhãn;
- tiền xử lý một mình không đủ để bảo đảm mô hình hoàn toàn không học shortcut.

Tài liệu này chuẩn hóa:

- `canonical evidence core`,
- `support gate` đúng về mặt hình học,
- metadata và logging bắt buộc,
- quy trình audit nuisance ở mức patch, feature và score.

Tài liệu này **không** chuẩn hóa:

- kiến trúc classifier,
- bộ đặc trưng cụ thể của các phiên bản tương lai,
- ngưỡng phân loại ở tầng sản phẩm.

## 2. Ký hiệu và đối tượng

### 2.1. Dữ liệu đầu vào

- `B`: byte-stream đầu vào của tệp ảnh.
- `D_k(B)`: phép giải mã bằng decoder stack `k` đã được pin version.
- `O`: phép hiệu chỉnh orientation ở mức pixel.
- `X_0 = O(D_k(B))`: ảnh sau decode và orientation.
- `H, W`: chiều cao và chiều rộng của `X_0`.
- `S = min(H, W)`: cạnh ngắn của `X_0`.

### 2.2. Biến ngẫu nhiên và nuisance

- `Y in {0,1}`: nhãn; `Y = 1` là ảnh AI, `Y = 0` là ảnh real.
- `Z`: nội dung/scene ẩn.
- `H_c`: lịch sử nén và vận chuyển ẩn trong ảnh quan sát được.
- `N_fmt`: format/container gốc của file (`JPEG`, `PNG`, ...).
- `N_sub`: chroma subsampling history.
- `N_codec`: decoder hoặc codec path.
- `N_size(t) = 1{S >= t}`: support gate tại ngưỡng `t`.

### 2.3. Tham số chuẩn của v3

- `r = (r_x, r_y) = (3, 3)`: residue cố định trên lattice `mod 8`.
- `C = 248`: kích thước crop chuẩn của main branch.
- `T = C + 3 = 251`: ngưỡng support tối thiểu để crop `C x C` tại residue `(3,3)` tồn tại thật.
- `A_x(W) = {x in Z : 0 <= x <= W - C, x = r_x (mod 8)}`.
- `A_y(H) = {y in Z : 0 <= y <= H - C, y = r_y (mod 8)}`.
- `c_x = (W - C) / 2`, `c_y = (H - C) / 2`: tâm hình học liên tục của crop.
- `x* = argmin_{x in A_x(W)} |x - c_x|`, nếu hòa thì chọn giá trị nhỏ hơn.
- `y* = argmin_{y in A_y(H)} |y - c_y|`, nếu hòa thì chọn giá trị nhỏ hơn.
- `X_can = X_0[y*:y*+C, x*:x*+C]`: canonical native crop.
- `T_col`: phép đổi màu từ BGR sang YCrCb đúng theo decoder stack đang dùng.

### 2.4. Metric audit

- `Phi(X) in R^d`: vector đặc trưng downstream.
- `s(X)`: raw score của classifier.
- `A_lab(.)`: ROC-AUC trên bài toán nhãn `Y`.
- `A_nuis(.)`: ROC-AUC trên một bài toán nuisance đã được cân bằng.
- `Delta_lab(j) = A_lab(Phi) - A_lab(Phi_{-j})`.
- `Delta_nuis(j) = A_nuis(Phi) - A_nuis(Phi_{-j})`.

## 3. Dữ liệu và số liệu đã xác thực lại

Các số liệu dưới đây được kiểm định lại trực tiếp từ:

- `audit_output/data_audit/metadata/per_file_metadata.parquet`
- `audit_output/validation/pipeline_revision/pipeline_revision_validation.json`
- `audit_output/validation/pipeline_revision/pipeline_revision_variant_metrics.csv`
- `models/03_artifacts/evaluation_metrics.csv`

### 3.1. Bất đối xứng container là tuyệt đối

Trên corpus hiện tại:

| Đại lượng | Giá trị |
|---|---:|
| `P(JPEG | real)` | `1.0000` |
| `P(PNG | ai)` | `1.0000` |

Hệ quả:

- nếu downstream được phép nhìn bytes/container, bài toán suy biến;
- decode là bước bắt buộc, không phải tùy chọn.

### 3.2. Pipeline cũ tạo shortcut hình học rất mạnh

Pipeline cũ dùng:

- `reflect pad if needed`,
- `conditional +3`,
- crop `256 x 256`.

Trên dữ liệu hiện tại:

| Đại lượng | AI | Real |
|---|---:|---:|
| `P(S < 259)` | `0.4091` | `0.0390` |
| `P(conditional_shift = 1)` | `1.0000` | `0.1361` |
| `P(S >= 259)` | `0.5909` | `0.9611` |

Mutual information của support gate cũ:

- `I(Y; 1{S >= 259}) = 0.1607 bit`.

Kết luận:

- `pad-if-needed`,
- `conditional +3`,
- và support gate cũ

đều là nguồn shortcut có thực, không còn là giả thuyết.

### 3.3. Thay `256` bằng `248` giải gần như toàn bộ bài toán coverage do hình học

Giữ residue `(3,3)` và thay crop size:

| Cấu hình | Ngưỡng support | `P(accepted | AI)` | `P(accepted | Real)` | `I(Y; accepted)` |
|---|---:|---:|---:|---:|
| `C = 256` | `259` | `0.5909` | `0.9611` | `0.1607 bit` |
| `C = 248` | `251` | `1.0000` | `0.9642` | `0.0181 bit` |
| `C = 240` | `243` | `1.0000` | `0.9673` | `0.0165 bit` |

Hệ quả trực tiếp của `C = 248`:

- không còn fake nào bị đẩy sang `LOW_SUPPORT` trên corpus hiện tại,
- chỉ còn khoảng `3.58%` ảnh real là low-support,
- mutual information của support gate giảm xấp xỉ `8.9x`,
- diện tích bằng chứng giữ lại là `248^2 / 256^2 = 0.9385`.

### 3.4. `C = 248` giữ được tất cả generator fake hiện có

Với ngưỡng `T = 251`:

| Generator fake | `P(accepted | generator, ai)` |
|---|---:|
| `adm` | `1.0000` |
| `glide` | `1.0000` |
| `midjourney` | `1.0000` |
| `sdv14` | `1.0000` |
| `sdv15` | `1.0000` |
| `vqdm` | `1.0000` |
| `wukong` | `1.0000` |

Đây là khác biệt cấu trúc quan trọng nhất giữa v2 và v3: support gate không còn cắt mất cả cụm generator `256 x 256`.

### 3.5. Chroma canonicalization không đủ tốt để làm core canonicalizer

Theo benchmark sạch hiện có:

| Variant | Task | AUC |
|---|---|---:|
| `native_nojpeg` | `real_vs_fake` | `0.9037` `LOGO AUC` |
| `native_chroma420` | `real_vs_fake` | `0.7726` `LOGO AUC` |
| `native_nojpeg` | `real_444_vs_420` | `0.9760` |
| `native_chroma420` | `real_444_vs_420` | `0.9674` |

Kết luận:

- `chroma420` làm giảm mạnh label utility,
- nhưng không làm nuisance subsampling biến mất ở mức đủ để dùng như canonicalizer chính.

### 3.6. Giới hạn hiện tại phải ghi rõ

Hiện chưa có benchmark đầy đủ của bộ `33` feature tại `248 x 248`, vì code extractor hiện tại đang khóa shape `(256, 256, 3)`.

Do đó:

- lựa chọn `C = 248` hiện được hỗ trợ trực tiếp bởi hình học và thống kê coverage,
- nhưng chưa được phép diễn giải là "đã chứng minh tăng AUC" cho pipeline cuối.

## 4. Các mệnh đề lý thuyết

### 4.1. Decode loại đúng shortcut ở mức bytes, không loại được history ở mức pixel

Nếu:

`O(D_k(B_1)) = O(D_k(B_2))`,

thì mọi downstream chỉ nhìn `X_0` đều không thể phân biệt `B_1` và `B_2`.

Đây là mệnh đề đúng cho:

- metadata,
- EXIF,
- ICC,
- container bytes.

Nhưng nếu ảnh quan sát có dạng:

`X_0 = T_{H_c}(Z)`,

trong đó `H_c` là compression history, thì decode không biến `H_c` thành rỗng. History vẫn nằm trong pixel.

### 4.2. Định lý shortcut do branch phụ thuộc dữ liệu

Xét một tiền xử lý có hai nhánh:

`P(X_0) = P_0(X_0)` nếu `N(X_0) = 0`,  
`P(X_0) = P_1(X_0)` nếu `N(X_0) = 1`.

Nếu:

1. `P(N = 1 | Y = 1) != P(N = 1 | Y = 0)`,
2. downstream có thể phân biệt ảnh sau `P_0` và `P_1`,

thì `N` là một shortcut khả dĩ.

Pipeline cũ vi phạm điều kiện này với:

- `N = need_pad`,
- `N = conditional_shift`,
- `N = 1{S >= 259}`.

### 4.3. Định lý tồn tại exact residue crop

Cho `C > 0`, `0 <= r_x, r_y < 8`.

Khi đó:

- `A_x(W)` khác rỗng khi và chỉ khi `W >= C + r_x`,
- `A_y(H)` khác rỗng khi và chỉ khi `H >= C + r_y`.

**Chứng minh.**

- Nếu `W >= C + r_x`, chọn `x = r_x`. Khi đó `0 <= x <= W - C` và `x = r_x (mod 8)`, nên `A_x(W)` khác rỗng.
- Nếu `W < C + r_x`, mọi `x = r_x + 8m` đều thỏa `x >= r_x > W - C`, nên không có phần tử hợp lệ. Suy ra `A_x(W)` rỗng.
- Lập luận tương tự cho `A_y(H)`. QED.

Hệ quả:

- với `C = 248`, `r = (3,3)`, crop tồn tại khi và chỉ khi `S >= 251`;
- ảnh `256 x 256` đủ support cho crop này mà không cần pad hay resize.

### 4.4. Fixed residue chỉ có nghĩa phase-fixing trên lattice hiện tại

Chọn residue cố định `(3,3)` chỉ bảo đảm:

- top-left của crop là tất định trong hệ tọa độ sau decode/orientation,
- không còn branch `if aligned then +3`.

Nó **không** bảo đảm:

- misalignment phổ quát với latent JPEG grid trước orientation,
- hay "phá" mọi DCT grid ẩn trong mọi ảnh.

Mệnh đề này bắt buộc phải ghi rõ để tránh overclaim.

### 4.5. Không tồn tại compression-history eraser tất định phổ quát

Với JPEG round-trip `J_q`, do lượng tử hóa là ánh xạ nhiều-một, tồn tại `x_1 != x_2` sao cho:

`J_q(x_1) = J_q(x_2)`.

Suy ra không tồn tại phép tất định `E` mà với mọi quan sát `J_q(x)` có thể:

1. khôi phục chính xác tiền thân "trước nén", hoặc
2. đưa mọi ảnh đã qua các lịch sử nén khác nhau về cùng một trạng thái "đã xóa history" mà vẫn giữ nguyên toàn bộ thông tin forensic hữu ích.

Hệ quả:

- `JPEG bottleneck` không được gọi là `history eraser`,
- `class-blind recompression` không được diễn giải thành phép loại bỏ latent history.

### 4.6. Operator-blindness không kéo theo distributional neutrality

Nếu `g` được chọn mà không dùng nhãn, ta chỉ có:

- `P(g | X, Y) = P(g)` hoặc ít nhất `P(g | Y) = P(g)`.

Điều đó **không** suy ra:

- `P(g(X) | Y = 0) = P(g(X) | Y = 1)`.

Nếu:

- `X = T_{H_c}(Z)`,
- `P(H_c | Y = 0) != P(H_c | Y = 1)`,

thì cùng một `g` vẫn có thể sinh ra hai phân phối đầu ra rất khác nhau theo lớp. Vì vậy, v3 không đưa bất kỳ `G_default` nào vào phần correctness claim của core preprocessing.

## 5. Tiêu chí chọn crop size chuẩn

Ta chọn `C` theo thứ tự ưu tiên sau:

1. `C` là bội số của `8`,
2. exact crop với residue `(3,3)` phải tồn tại trên ảnh `256 x 256`,
3. `C` lớn nhất có thể để giữ lại nhiều bằng chứng nhất,
4. support gate do `C` sinh ra không được gây sụp coverage cho lớp AI trên corpus hiện tại.

Từ điều kiện (2):

- cần `C + 3 <= 256`.

Trong tập các bội số của `8`, giá trị lớn nhất là:

- `C* = 248`.

Vì vậy, `248` là nghiệm tối ưu theo tiêu chí từ điển:

- giữ area lớn nhất,
- đồng thời xóa được fake coverage collapse do crop `256 @ residue 3`.

## 6. Canonical evidence core của v3

### 6.1. Đầu vào và đầu ra

Đầu vào:

- byte-stream `B`.

Đầu ra:

- `status in {ACCEPTED, LOW_SUPPORT, DECODE_ERROR, UNSUPPORTED_FORMAT}`,
- nếu `status = ACCEPTED`, trả về `X_can_ycrcb in uint8^{248 x 248 x 3}`,
- luôn trả về metadata audit bắt buộc.

### 6.2. Bước 1: decode và orientation

Sinh:

- `X_0 = O(D_k(B))`.

Yêu cầu:

- dùng một decoder stack duy nhất cho train, validation, test và deploy;
- pin version của decoder;
- không branch theo label, filename, đường dẫn, generator, hay format;
- orientation được sửa ở mức pixel;
- nếu có alpha channel thì việc loại bỏ alpha phải theo đúng quy tắc cố định của decoder stack.

Nếu decode thất bại:

- trả `DECODE_ERROR`.

### 6.3. Bước 2: support gate

Tính:

- `S = min(H, W)`.

Nếu `S < 251`:

- trả `LOW_SUPPORT`,
- không được pad,
- không được resize,
- không được nội suy để "đủ chuẩn" rồi đưa vào main branch.

Nếu `S >= 251`:

- tiếp tục bước crop.

### 6.4. Bước 3: exact residue center crop

Tính:

- `A_x(W) = {x : 0 <= x <= W - 248, x = 3 (mod 8)}`,
- `A_y(H) = {y : 0 <= y <= H - 248, y = 3 (mod 8)}`,
- `x* = argmin_{x in A_x(W)} |x - (W - 248)/2|`,
- `y* = argmin_{y in A_y(H)} |y - (H - 248)/2|`.

Nếu có hai ứng viên cách tâm bằng nhau:

- chọn giá trị nhỏ hơn.

Sinh:

- `X_can = X_0[y*:y*+248, x*:x*+248]`.

Tính chất:

- tất định,
- không tạo pixel mới,
- không còn `conditional +3`,
- phase của crop được cố định trong hệ tọa độ hiện tại.

### 6.5. Bước 4: đổi màu

Sinh:

- `X_can_ycrcb = T_col(X_can)`.

Yêu cầu:

- quy ước kênh phải giống nhau giữa train và infer;
- nếu dùng OpenCV thì phải thống nhất chính xác chuẩn `YCrCb` mà runtime đang dùng;
- downstream không được lấy metadata bytes-level làm feature phân lớp mặc định.

### 6.6. Các phép bị cấm trong core

Những phép sau bị cấm tuyệt đối trong main preprocessing core:

- `reflect padding`,
- `zero padding`,
- `replicate padding`,
- `resize to 256`,
- `resize short side to 263`,
- `conditional +3`,
- `always +3` như một cách thay thế `conditional +3`,
- `deterministic JPEG bottleneck`,
- `random JPEG bottleneck`,
- `chroma420 canonicalization`,
- `gray-only canonicalization`.

Lý do:

- hoặc chúng tạo pixel không tồn tại trong ảnh quan sát,
- hoặc chúng mang claim quá mạnh mà chưa chứng minh được,
- hoặc thực nghiệm hiện có cho thấy trade-off label-vs-nuisance không đủ tốt để đưa vào core.

## 7. Chính sách `LOW_SUPPORT`

`LOW_SUPPORT` trong v3 không còn cắt mất `40.91%` ảnh AI như ở pipeline cũ, nhưng vẫn là trạng thái cần giữ để bảo toàn tính trung thực của evidence.

Yêu cầu bắt buộc:

1. `LOW_SUPPORT` là trạng thái routing/abstention, không phải classifier branch của main model.
2. Mọi metric của main branch phải được báo cáo trên tập `ACCEPTED`.
3. Nếu sau này có low-support head riêng, nó phải có benchmark và đặc tả riêng; không được trộn metric của nó với main branch mà không báo cáo coverage.

## 8. Vì sao v3 không đưa degradation suite vào core

### 8.1. Bài toán gốc là latent history, không chỉ là transform sau cùng

Giả sử:

`X_0 = T_{H_c}(Z)`.

Nếu sau đó áp thêm `g`, ta có:

`g(X_0) = g(T_{H_c}(Z))`.

Khi đó:

- `E_g[s(g(X_0))]`

chỉ marginalize được biến ngẫu nhiên `g`, không marginalize được latent history `H_c` đã nằm sẵn trong `X_0`.

### 8.2. Hệ quả chuẩn hóa

V3 không định nghĩa `G_default` cho training hay inference của core.

Các phép như:

- `JPEG(Q)`,
- `resize`,
- `chroma420 projection`

chỉ có thể xuất hiện dưới vai trò:

- `stress test`,
- `audit suite`,
- hoặc augmentation có điều kiện sau khi đã có telemetry production phù hợp.

Mọi kết luận từ các phép này phải được diễn giải là:

- `empirical transport robustness`,

không được diễn giải là:

- `compression-history invariance`.

## 9. Quy trình audit bắt buộc

Audit là một phần của đặc tả, không phải phụ lục tùy chọn.

### 9.1. Patch-level audit

Phải báo cáo ít nhất:

1. `A_nuis_patch(N_size(251))` trên corpus raw;
2. `A_nuis_patch(N_sub)` trên benchmark real-only cân bằng `4:4:4` vs `4:2:0`;
3. nếu thay decoder, `A_nuis_patch(N_codec)` trên benchmark decode-matched.

Mục tiêu của patch audit không phải ép mọi nuisance AUC về `0.5` bằng biến đổi phá dữ liệu, mà là phát hiện pipeline nào làm nuisance dễ học hơn.

### 9.2. Feature governance đúng nghĩa

Không được dùng ngưỡng marginal AUC tùy ý để drop feature.

Quy trình đúng:

1. tính `Delta_lab(j)` bằng leave-one-feature-out hoặc leave-one-group-out;
2. tính `Delta_nuis(j)` trên đúng benchmark nuisance;
3. dùng bootstrap để lấy khoảng tin cậy cho `Delta_lab(j)` và `Delta_nuis(j)`.

Feature `j` bị loại nếu đồng thời:

- cận trên của CI cho `Delta_lab(j)` `<= 0`,
- cận dưới của CI cho `Delta_nuis(j)` `> 0`.

Đây là rule có điều kiện; nó tránh sai lầm của marginal screening.

### 9.3. Score-level audit

Phải báo cáo:

- `A_nuis_score = A_nuis(s(X_can))`

trên ít nhất các benchmark sau:

1. `real JPEG 4:4:4 vs 4:2:0`,
2. `accepted vs low_support` nếu về sau xuất hiện low-support head,
3. benchmark theo `decoder_id` nếu decoder thay đổi.

Nếu về sau thêm bất kỳ stress transform nào `g`, phải báo cáo đồng thời:

- `A_nuis(s(X))`,
- `A_nuis(s(g(X)))`,
- `A_nuis(E_g[s(g(X))])`.

Không được mô tả `E_g[s(g(X))]` là đã xóa latent history nếu không có benchmark đối chứng hỗ trợ phát biểu đó.

## 10. Các phương án đã bị loại

### 10.1. `256 + reflect pad + conditional +3`

Bị loại vì:

- `P(need_pad | AI) = 0.4091` so với `0.0390` ở real;
- `P(conditional_shift | AI) = 1.0` so với `0.1361` ở real;
- `I(Y; 1{S >= 259}) = 0.1607 bit`.

### 10.2. `resize263`

Bị loại vì:

- resize là phép mất thông tin,
- không có cơ sở để làm canonicalizer chính,
- benchmark hiện có cho thấy `resize263_*` kém hơn native variants ở label task.

### 10.3. `chroma420` trong core

Bị loại vì:

- `LOGO AUC` giảm mạnh (`0.9037 -> 0.7726`),
- nuisance `444 vs 420` vẫn rất cao (`0.9760 -> 0.9674`).

### 10.4. `JPEG bottleneck` trong core

Bị loại vì:

- không có chứng minh xóa history,
- có thể tạo cue mới,
- và theo lý thuyết chỉ là thêm một nuisance transform nữa.

## 11. Yêu cầu triển khai

### 11.1. Train/infer parity

Những thành phần sau phải tuyệt đối đồng nhất giữa train và infer:

- decoder stack,
- orientation rule,
- crop rule,
- color transform,
- dtype đầu vào của feature extractor.

### 11.2. Yêu cầu kỹ thuật riêng của repo hiện tại

Để đưa v3 vào train thực sự, cần sửa đồng bộ:

- validator shape `(256, 256, 3)` trong các extractor,
- mọi hằng số phụ thuộc `CROP_SIZE`,
- mọi giả định ngầm về `256 x 256` ở tầng feature và inference runtime.

Nếu không làm bước này, v3 mới chỉ là đặc tả hình học đúng, chưa phải pipeline chạy được end-to-end.

### 11.3. Logging

Mỗi artifact train và mỗi request infer phải log:

- `decoder_id`,
- `decoder_version`,
- `crop_size = 248`,
- `crop_residue = (3,3)`,
- `support_threshold = 251`,
- `status`,
- `crop_x0`, `crop_y0`,
- `raw_width`, `raw_height`.

### 11.4. Versioning

Nếu thay đổi một trong các thành phần sau:

- decoder,
- crop size,
- residue,
- color transform,

thì phải tăng preprocessing spec version và retrain từ đầu.

## 12. Kết luận chuẩn hóa

Trong không gian giải pháp chỉ được phép can thiệp vào tiền xử lý, pipeline tối ưu và trung thực nhất hiện tại là:

`decode -> orientation -> exact residue center crop 248@3,3 -> YCrCb`

với các nguyên tắc bất di bất dịch:

- không pad,
- không resize,
- không recompress trong core,
- không dùng `class-blind transform` như một bằng chứng giả về invariance,
- không báo cáo chung metric của `LOW_SUPPORT` với main branch khi chưa có head riêng.

Phán quyết cuối cùng của v3 là:

1. nó loại bỏ được các shortcut hình học lớn nhất đã được xác nhận;
2. nó giảm cực mạnh support leakage mà không phải bịa pixel;
3. nó không overclaim về khả năng xóa compression history;
4. nó đặt phần rủi ro còn lại đúng vị trí: audit, feature governance và model evaluation.
