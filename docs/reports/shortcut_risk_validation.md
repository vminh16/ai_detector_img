# Báo cáo rủi ro shortcut và tái thẩm định pipeline

## 1. Mục tiêu

Tài liệu này thay thế toàn bộ các báo cáo rời rạc trước đây về:

- confound do `reflect padding`,
- confound do `conditional +3 grid misalignment`,
- confound do branch theo kích thước,
- rủi ro do lịch sử JPEG / chroma subsampling,
- vai trò thực sự của resize, JPEG bottleneck và chroma canonicalization.

Mục tiêu của tài liệu là:

1. phát biểu lại bài toán bằng ký hiệu toán học rõ ràng,
2. tách phần nào được chứng minh chắc chắn khỏi phần nào chỉ được hỗ trợ bởi thực nghiệm,
3. kiểm định lại các claim quan trọng bằng số liệu hiện có trong repo,
4. đưa ra phán quyết cuối cùng cho policy preprocessing của repo.

Artifact nội bộ được dùng trong báo cáo này:

- `audit_output/data_audit/metadata/per_file_metadata.parquet`
- `data/processed/manifest.csv`
- `features/features_dataset.csv`
- `models/03_artifacts/evaluation_metrics.csv`
- `audit_output/validation/pipeline_revision/pipeline_revision_validation.json`
- `audit_output/validation/pipeline_revision/pipeline_revision_variant_metrics.csv`
- `audit_output/validation/pipeline_revision/pipeline_revision_feature_subset_metrics.csv`
- `audit_output/validation/risk_solution/risk_solution_validation.json`
- `audit_output/validation/risk_solution/risk_solution_variant_metrics.csv`

## 2. Ký hiệu và quy ước

### 2.1. Biến ngẫu nhiên

- `Y in {0,1}`: nhãn, với `Y = 1` là ảnh AI và `Y = 0` là ảnh real.
- `B`: byte-stream đầu vào của file ảnh.
- `D(B)`: phép giải mã bytes thành tensor pixel.
- `O`: phép sửa orientation theo EXIF ở mức pixel.
- `X0 = O(D(B))`: ảnh sau decode và orientation.
- `H, W`: chiều cao và chiều rộng của `X0`.
- `S = min(H, W)`: cạnh ngắn.
- `X1`: patch sau bước crop trên native lattice.
- `Phi(X1)`: vector feature downstream.
- `f(Phi(X1))`: score hoặc logit/probability do model sinh ra.

### 2.2. Biến nuisance

- `N_fmt`: format/container gốc của file (`JPEG`, `PNG`, ...).
- `N_pad`: cờ padding, bằng `1` nếu ảnh phải pad để đủ support.
- `N_shift`: cờ branch `conditional +3`.
- `N_size(t)`: cờ branch kích thước, bằng `1` nếu `S >= t`.
- `N_sub`: lịch sử chroma subsampling của real JPEG (`4:4:4`, `4:2:0`, ...).

### 2.3. Hai loại kết luận

- **Kết luận định lý / mệnh đề**: đúng trong lớp giả thiết đã phát biểu.
- **Kết luận thực nghiệm**: đúng trên dữ liệu và protocol hiện có trong repo, không được diễn giải thành định luật phổ quát ngoài phạm vi đó.

## 3. Sự kiện dữ liệu đã kiểm định lại

Từ `audit_output/data_audit/metadata/per_file_metadata.parquet`:

| Đại lượng | Giá trị |
|---|---:|
| Tổng số ảnh hợp lệ | `88,007` |
| Real | `44,005` |
| Fake | `44,002` |
| Generator fake | `adm`, `glide`, `midjourney`, `sdv14`, `sdv15`, `vqdm`, `wukong` |

### 3.1. Bất đối xứng định dạng là tuyệt đối

| Xác suất | Giá trị |
|---|---:|
| `P(JPEG | nature)` | `1.0000` |
| `P(PNG | ai)` | `1.0000` |

Điều này có nghĩa:

- nếu downstream được phép nhìn bytes/container hoặc metadata, bài toán gần như suy biến,
- decode là bước bắt buộc, không phải tùy chọn.

### 3.2. Bất đối xứng hình học là rất mạnh

Với ngưỡng support hình học đúng cho crop `256 x 256` ở residue `(3,3)`:

- `S >= 259` là ảnh **đủ support**,
- `S < 259` là ảnh **low-support**.

Thống kê từ raw dataset:

| Đại lượng | AI | Real |
|---|---:|---:|
| `P(N_pad = 1)` với ngưỡng `259` | `0.4091` | `0.0390` |
| `P(N_shift = 1)` trong pipeline cũ | `1.0000` | `0.1361` |
| `P(S >= 263)` | `0.5909` | `0.9598` |
| `P(S >= 259)` | `0.5909` | `0.9611` |

Mutual information giữa branch kích thước và nhãn:

- `I(Y; N_size(263)) = 0.1590 bit`
- `I(Y; N_size(259)) = 0.1607 bit`

Nói cách khác:

- branch kích thước mang thông tin nhãn rõ rệt,
- dùng nó như một classifier branch đồng hạng với main branch là không an toàn.

### 3.3. Cấu trúc kích thước bám chặt theo generator

Với ngưỡng `259` hoặc `263`, các generator fake chia thành hai nhóm gần như rời nhau:

| Generator fake | `P(S >= 259)` |
|---|---:|
| `adm` | `0.0000` |
| `glide` | `0.0000` |
| `vqdm` | `0.0000` |
| `midjourney` | `1.0000` |
| `sdv14` | `1.0000` |
| `sdv15` | `1.0000` |
| `wukong` | `1.0000` |

Do đó, mọi kiến trúc `main branch / small branch` xây trực tiếp từ raw size sẽ gần như học generator structure của fake.

### 3.4. Lịch sử chroma subsampling của real JPEG là có thật

Từ `audit_output/validation/risk_solution/risk_solution_validation.json`, phân bố subsampling của ảnh real JPEG trên disk:

| Subsampling | Count | Rate |
|---|---:|---:|
| `4:4:4` | `38,024` | `0.8646` |
| `4:2:0` | `4,703` | `0.1069` |
| `4:2:2` | `431` | `0.0098` |
| `other` | `821` | `0.0187` |

Hệ quả:

- leakage về chroma bandwidth là một nuisance thực,
- nhưng nó không phải shortcut duy nhất trong repo này.

## 4. Phân tích toán học các shortcut chính

## 4.1. Decode triệt tiêu đúng shortcut container

### Mệnh đề 1

Giả sử hai file `B1`, `B2` khác nhau ở:

- metadata,
- EXIF,
- ICC,
- container bytes,
- encoder string,

nhưng sau giải mã và sửa orientation chúng cho cùng pixel:

`O(D(B1)) = O(D(B2))`.

Khi đó mọi pipeline downstream chỉ nhìn `X0 = O(D(B))` đều không thể phân biệt `B1` và `B2`.

### Ý nghĩa

Đây là một triệt tiêu **đúng**, không phải gần đúng:

- bytes-level shortcut bị cắt khỏi không gian quan sát,
- nhưng pixel-domain shortcut do nén, subsampling hay hình học vẫn còn.

## 4.2. Mọi transform có branch phụ thuộc dữ liệu đều là nguồn shortcut tiềm năng

Xét một preprocessing dạng nhị phân:

`T(X0) = T0(X0)` nếu `N(X0) = 0`, và `T(X0) = T1(X0)` nếu `N(X0) = 1`.

Giả sử:

1. `P(N = 1 | Y = 1) != P(N = 1 | Y = 0)`,
2. `T0(X0)` và `T1(X0)` không cùng phân phối dưới một representation downstream nào đó.

Khi đó `N` là một shortcut khả dĩ: classifier có thể khai thác khác biệt giữa hai nhánh thay vì học dấu vết forensic nội tại.

### Áp dụng cho repo này

- `N = N_pad` với nhánh `reflect pad if needed`,
- `N = N_shift` với nhánh `if aligned then +3`,
- `N = N_size(t)` với branch theo kích thước.

Ba biến này đều thỏa điều kiện (1) rất mạnh trên dataset hiện tại.

## 4.3. Vì sao `conditional +3` là confound nghiêm trọng

Trong pipeline cũ:

- gốc crop trung tâm là `(x0, y0)`,
- nếu `x0` hoặc `y0` rơi vào residue `0 mod 8` trong hệ tọa độ ảnh gốc, pipeline cộng thêm `+3`.

Đặt:

`N_shift = 1{ x0 = 0 mod 8 or y0 = 0 mod 8 }`.

Trên raw dataset hiện tại:

- `P(N_shift = 1 | ai) = 1`,
- `P(N_shift = 1 | nature) = 0.1361`.

Lý do hình học:

- fake chủ yếu có kích thước `256`, `512`, `1024`, nên center crop thường bắt đầu ở `0`, `128`, `384`,
- các giá trị này đều là `0 mod 8`,
- do đó fake gần như luôn bị đẩy qua nhánh `+3`.

Kết luận:

- `conditional +3` không phải một phép “scrub JPEG grid” trung tính,
- nó là một transform phụ thuộc dữ liệu gốc, bị lệch mạnh theo nhãn.

## 4.4. Vì sao `reflect padding` không được phép nằm trong main branch

Với `S < 259`, crop `256 x 256` ở residue `(3,3)` không tồn tại trên native lattice.

Nếu ta vẫn ép ảnh vào main branch bằng:

- reflect padding,
- zero padding,
- resize,
- hoặc nội suy bất kỳ,

thì patch đầu ra không còn là một crop thuần từ pixel quan sát được nữa.

### Mệnh đề 2

Nếu `S < 259`, mọi phép sinh tensor `256 x 256` cho main branch đều phải dùng:

- extrapolation,
- interpolation,
- hoặc cả hai.

Đó là các phép xác định `G` sao cho đầu ra chứa pixel không quan sát trực tiếp từ ảnh gốc.

Về mặt information theory:

`I(Y; G(X0)) <= I(Y; X0)`.

Nhưng quan trọng hơn với repo này:

- do `P(S < 259 | ai)` và `P(S < 259 | real)` khác nhau mạnh,
- branch “có pad / không pad” trở thành một nuisance cực mạnh,
- feature extractor sau đó có thể học chính biên phản xạ hoặc hệ quả thống kê của nó.

## 4.5. Không tồn tại phép xác định nào xóa chính xác lịch sử JPEG chưa biết

Gọi `J_q` là phép round-trip JPEG với quality `q`.

Do lượng tử hóa DCT là ánh xạ nhiều-một, tồn tại `x1 != x2` sao cho:

`J_q(x1) = J_q(x2)`.

### Mệnh đề 3

Không tồn tại một phép xác định `T` sao cho `T(J_q(x))` “xóa chính xác lịch sử JPEG trước đó” cho mọi `x` và mọi `q` chưa biết.

### Hệ quả

- thêm một JPEG bottleneck không thể là một “history eraser” đúng nghĩa,
- nó chỉ áp thêm một phép mất mát mới lên ảnh hiện tại.

Điều này **không** có nghĩa JPEG bottleneck luôn vô ích trong mọi benchmark; nó chỉ có nghĩa:

- ta không được gán cho nó năng lực toán học mà nó không có,
- và không được dùng nó như canonicalizer chính rồi tuyên bố bài toán compression history đã được giải.

## 4.6. Resize toàn cục không thể là canonicalizer tối ưu

Với mọi deterministic transform `R`:

`I(Y; R(X0)) <= I(Y; X0)`.

Đây là bất đẳng thức xử lý dữ liệu.

### Hệ quả đúng

- resize không thể tạo thêm thông tin forensic mới,
- tốt nhất nó giữ lại đủ; tệ hơn nó làm mất thông tin.

### Điều không được suy diễn quá mức

Bất đẳng thức trên **không** tự động kéo theo rằng mọi finite-sample AUC của mọi model thực dụng đều phải giảm.  
Điều được kết luận từ repo này là:

1. về mặt lý thuyết, resize không có cơ sở để trở thành canonicalizer chính,
2. về mặt thực nghiệm của repo, resize làm yếu tín hiệu thật nhưng leakage vẫn còn cao.

## 4.7. Chroma low-band canonicalization chỉ là scrubber yếu, không phải lời giải trọn vẹn

Xét một toán tử tuyến tính class-independent trên chroma:

`L = U2 D2 H`

trong đó:

- `H` là low-pass separable,
- `D2` là decimate hệ số `2`,
- `U2` là upsample với quy ước cố định.

### Điều có thể khẳng định

- `L` là tuyến tính và class-independent,
- `I(Y; L(C)) <= I(Y; C)`,
- nó làm giảm mismatch về chroma bandwidth giữa `4:4:4` và `4:2:0`.

### Điều không được khẳng định

- `L` không xóa toàn bộ lịch sử nén JPEG,
- `L` không triệt toàn bộ subsampling leakage nếu feature space còn nhạy với residual/statistics khác ngoài high-band chroma,
- `L` không phải một “phép chiếu trực giao tối ưu” trừ khi implementation thật sự là projector theo nghĩa Hilbert. Pipeline nghiên cứu hiện tại dùng một operator thực dụng, không phải projector đúng nghĩa.

Kết luận chính xác hơn:

- chroma canonicalization chỉ là **một scrubber phụ trợ yếu nhưng hợp lý**,
- không được xem là lời giải đủ cho compression nuisance.

## 5. Kiểm định trực tiếp trên model và feature space hiện tại

## 5.1. Model hiện tại đã học shortcut hình học

Champion model hiện tại là LightGBM + Platt calibration. Trên tập **real-only**, score của model thay đổi rõ rệt theo các flag nuisance của preprocessing hiện hành.

### Toàn bộ real set

| Flag | Group | N | Mean score | High-rate tại `tau_op` |
|---|---|---:|---:|---:|
| `padded` | `False` | `42,267` | `0.2522` | `2.53%` |
| `padded` | `True` | `1,712` | `0.3088` | `3.39%` |
| `misalign_trigger` | `False` | `37,993` | `0.2447` | `2.16%` |
| `misalign_trigger` | `True` | `5,986` | `0.3165` | `5.13%` |

### Theo split đánh giá

| Split | `misalign=False` high-rate | `misalign=True` high-rate |
|---|---:|---:|
| `val` | `4.09%` | `10.71%` |
| `id_test` | `4.64%` | `9.93%` |
| `ood_eval` | `4.38%` | `10.84%` |

Kết luận:

- shortcut hình học không chỉ tồn tại trên raw data,
- nó đang tác động trực tiếp lên FPR của real images.

## 5.2. Shortcut đã ngấm vào feature space 33 chiều

Trên **real-only**, dùng logistic regression đơn giản để dự đoán nuisance từ 33 feature hiện tại:

| Bài toán nuisance | AUC |
|---|---:|
| Dự đoán `misalign_trigger` | `0.6950` |
| Dự đoán `needs_pad` | `0.7842` |

Điều này cho thấy:

- leakage không còn là biến “ngoài model”,
- nó đã được mã hóa trong representation downstream.

## 5.3. Clean counterfactual compression experiment

Tập sạch:

- `800` ảnh,
- 4 generator fake không cần pad: `Midjourney`, `SDv14`, `SDv15`, `Wukong`,
- `100 real + 100 fake` cho mỗi generator,
- loại toàn bộ ảnh cần pad.

So sánh các biến thể:

| Variant | `sum |d|` | `logreg_cv_auc` |
|---|---:|---:|
| `method_a_allshift` | `7.678` | `0.7915` |
| `pil_only_conditional` | `5.900` | `0.7540` |
| `pil_only_allshift` | `5.848` | `0.7574` |
| `baseline_allshift_cv2` | `5.551` | `0.7611` |
| `baseline_conditional_cv2` | `5.542` | `0.7647` |
| `method_b_conditional` | `5.508` | `0.7644` |
| `method_b_allshift` | `5.431` | `0.7686` |

Kết luận thực nghiệm đúng phạm vi:

- sau khi gỡ geometry confound, `Method B` không còn là tốt nhất,
- `Method A` cho separability cao nhất trên benchmark sạch hiện có,
- do đó lập luận cũ “Method B là tối ưu” không còn đứng vững.

## 5.4. Pipeline-only ablation trên benchmark large-image sạch

Tập large-image sạch:

- mọi ảnh có `S >= 263`,
- 4 generator fake không cần pad,
- `100 real + 100 fake` mỗi generator.

### Real-vs-fake

| Variant | Stratified AUC | LOGO AUC |
|---|---:|---:|
| `native_nojpeg` | `0.9318` | `0.9037` |
| `native_chroma420` | `0.8020` | `0.7726` |
| `resize263_nojpeg` | `0.8644` | `0.8243` |
| `resize263_chroma420` | `0.7723` | `0.7605` |

### Real-only `4:4:4` vs `4:2:0`

| Variant | Stratified AUC |
|---|---:|
| `native_nojpeg` | `0.9760` |
| `native_chroma420` | `0.9674` |
| `resize263_nojpeg` | `0.9343` |
| `resize263_chroma420` | `0.8617` |

### Diễn giải đúng mức

- `native_nojpeg` cho AUC nhãn rất cao, nhưng đồng thời dự đoán subsampling gần như hoàn hảo; đó là dấu hiệu shortcut mạnh.
- `native_chroma420` giảm leakage một phần, nhưng vẫn để lại AUC nuisance rất cao.
- `resize` làm yếu leakage một phần, nhưng đồng thời làm yếu luôn tín hiệu phân biệt.

## 5.5. Leakage không chỉ nằm ở `dct_mid_*`

Kết quả subset audit cho bài toán real-only `4:4:4` vs `4:2:0`:

| Variant | `all33` | `no_dct` | `no_freq` | `cms` |
|---|---:|---:|---:|---:|
| `native_nojpeg` | `0.9777` | `0.9763` | `0.9772` | `0.9772` |
| `native_chroma420` | `0.9650` | `0.9615` | `0.9623` | `0.9623` |
| `resize263_nojpeg` | `0.9355` | `0.9312` | `0.9373` | `0.9373` |
| `resize263_chroma420` | `0.8643` | `0.8626` | `0.8660` | `0.8660` |

Kết luận:

- leakage không tập trung riêng ở `dct_mid_*`,
- bỏ DCT không đủ,
- muốn giảm shortcut thật sự phải thay đổi cả preprocessing lẫn feature space.

## 6. Phán quyết cuối cùng cho preprocessing policy

### 6.1. Những gì phải bác bỏ

**Bác bỏ cứng** trong main branch:

1. `conditional +3`
2. `reflect padding` để giả lập ảnh low-support thành ảnh main-branch
3. `main / small branch` dựa trực tiếp trên raw size
4. `universal resize` như canonicalizer chính
5. `deterministic JPEG bottleneck` như canonicalizer chính

### 6.2. Những gì được giữ

**Giữ cứng**:

1. decode bytes -> pixel
2. orientation correction ở mức pixel
3. fixed-residue native crop
4. `LOW_SUPPORT` như trạng thái abstain/routing, không phải classifier branch

**Giữ có điều kiện**:

5. chroma low-band canonicalization `L = U2 D2 H`

Điều kiện:

- phải được mô tả đúng là scrubber yếu, không phải lời giải đầy đủ,
- phải audit lại nuisance predictability sau khi thêm hoặc bỏ nó,
- không được diễn giải quá mức bằng định lý projector nếu implementation không phải projector.

## 7. Những gì preprocessing không thể hứa

Pipeline preprocessing, dù được sửa đúng, vẫn **không thể** tự mình bảo đảm:

1. xóa sạch lịch sử JPEG cũ,
2. làm ảnh low-support tương đương ảnh accepted,
3. triệt toàn bộ subsampling leakage nếu feature space phía sau còn nhạy với nuisance,
4. đạt SLA sản phẩm chỉ với 33 handcrafted features hiện tại.

Điều này nhất quán với `models/03_artifacts/evaluation_metrics.csv`:

- ở `FPR ~= 5%`, `TPR` hiện chỉ quanh `42%` trên `val`,
- đây không còn là lỗi tuning nhỏ; đó là giới hạn của representation hiện tại.

## 8. Kết luận

Kết luận khoa học cuối cùng cho repo này là:

1. chẩn đoán “repo đang học shortcut nguy hiểm” là đúng;
2. hai shortcut hình học mạnh nhất là `reflect padding` và `conditional +3`;
3. branch theo kích thước là một confound rõ rệt và chỉ được dùng như selective abstention;
4. JPEG bottleneck và resize không phải canonicalizer chính hợp lý;
5. chroma canonicalization chỉ là scrubber phụ trợ;
6. lời giải đúng phải là:
   - geometry-first cleanup ở preprocessing,
   - sau đó feature-space redesign và nuisance audit có kiểm soát.

Đó là policy duy nhất hiện có thể bảo vệ vừa bằng lý thuyết, vừa bằng số liệu của repo này.
