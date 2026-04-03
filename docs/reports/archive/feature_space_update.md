# Báo cáo mở rộng feature space sau khi dọn geometry confound

## 1. Mục tiêu

Sau khi báo cáo rủi ro kết luận rằng:

- preprocessing hiện tại đang tạo shortcut hình học mạnh,
- cleanup đúng phải bắt đầu từ geometry-first policy,

câu hỏi tiếp theo là:

> Sau khi bỏ các confound lớn khỏi preprocessing, nên mở rộng feature space thủ công theo hướng nào để tăng generalization mà không đẩy repo quay lại shortcut cũ?

Tài liệu này tập trung vào câu hỏi đó. Nó không cố chứng minh một detector phổ quát cho mọi generator và mọi điều kiện web; mục tiêu của nó là:

1. chẩn đoán giới hạn của baseline 33 feature,
2. chọn các họ feature mới có cơ sở vật lý hoặc thống kê đủ chặt,
3. kiểm định các họ đó trên benchmark sạch hơn về geometry,
4. rút ra thứ tự ưu tiên triển khai thực tế.

Artifact nội bộ được dùng:

- `audit_output/studies/update_feature/update_feature_benchmark.json`
- `audit_output/studies/update_feature/update_feature_table.csv`
- `audit_output/studies/update_feature/update_feature_individual_metrics.csv`
- `audit_output/validation/risk_solution/risk_solution_variant_metrics.csv`

## 2. Ký hiệu

- `X1`: patch `256 x 256` sau preprocessing geometry-safe.
- `Y, Cr, Cb`: ba kênh của `X1` trong không gian `YCrCb`.
- `Phi_base(X1) in R^33`: vector baseline 33 feature hiện tại.
- `Phi_new(X1)`: khối feature mới được đề xuất.
- `Phi_union = [Phi_base, Phi_new]`: vector hợp nhất.
- `AUC_LOGO`: AUC leave-one-generator-out, là metric chính trong báo cáo này.

Tại sao dùng `AUC_LOGO` làm metric chính:

- pooled stratified CV dễ bị “ăn may” nhờ overlap generator statistics,
- bài toán của repo là generalization cross-generator,
- do đó LOGO phản ánh mục tiêu sản phẩm tốt hơn.

## 3. Chẩn đoán baseline 33 feature

Baseline hiện tại gồm 4 nhóm:

1. frequency
2. color
3. microtexture
4. spatial

Nó có ba giới hạn cấu trúc.

### 3.1. Thiếu dấu vết hình thành ảnh của camera

Baseline hiện tại chủ yếu đo:

- phổ Fourier / DCT,
- tương quan màu cấp thấp,
- residual vi mô,
- một số ratio không gian.

Nó chưa đo trực diện các cơ chế vật lý quan trọng của ảnh chụp:

- periodicity do CFA / demosaicing,
- luật phụ thuộc giữa mức sáng và phương sai nhiễu,
- phụ thuộc đa tỉ lệ giữa hệ số wavelet.

Nói ngắn gọn:

- baseline nghiêng về “artifact summary”,
- nhưng chưa chạm đúng vào “image formation trace”.

### 3.2. Một phần baseline vẫn nhạy với compression nuisance

Báo cáo rủi ro đã cho thấy:

- `dct_mid_*` có rủi ro cao vì bám lưới JPEG / residual cục bộ,
- `cross_noise_ratio`, một phần chroma residual và spatial stats vẫn phản ứng với subsampling history,
- leakage không chỉ nằm ở một feature đơn lẻ.

Điều này không làm baseline vô dụng; nó chỉ có nghĩa:

- nếu muốn tăng generalization, ta cần bổ sung tín hiệu **trực giao hơn**,
- không nên tiếp tục “thêm nhiều heuristic texture tương tự nhau”.

### 3.3. Baseline đã chạm trần trên benchmark sạch hơn

Trên benchmark sạch bảo thủ `pil_only_allshift`, kết quả của baseline 33:

- `stratified_auc = 0.7593`
- `logo_auc = 0.7377`

Mức này đủ cho research baseline, nhưng chưa đủ headroom cho mục tiêu sản phẩm nghiêm ngặt.

## 4. Các họ feature ứng viên

Báo cáo này đánh giá ba họ feature mới:

1. periodicity do CFA,
2. noise-level function,
3. phụ thuộc đa tỉ lệ kiểu wavelet.

Tiêu chí lựa chọn:

- có mô hình toán học rõ ràng,
- chi phí CPU thấp,
- có cơ hội tạo thông tin trực giao với baseline 33.

## 4.1. Họ A: periodicity do CFA / demosaicing

### 4.1.1. Động cơ vật lý

Camera Bayer không đo đầy đủ RGB tại mọi pixel. Nó chỉ quan sát một màu trên mỗi vị trí rồi nội suy các màu còn lại. Quá trình này sinh ra các tương quan chu kỳ `2 x 2` trong các kênh màu.

Ngược lại, ảnh diffusion phổ biến được sinh trực tiếp thành tensor RGB đầy đủ. Nó không buộc phải đi qua cơ chế lấy mẫu CFA.

Kết quả là:

- periodicity chu kỳ `2 x 2` là tín hiệu có ý nghĩa vật lý đối với ảnh chụp,
- nếu tồn tại bền vững trong dữ liệu web, đó là trục rất hứa hẹn để phân biệt real và fake.

### 4.1.2. Mô hình toán học

Xét một residual thông cao `R` trên chroma sau khi dịch tâm. Trên lattice chẵn, định nghĩa ba hàm cơ sở zero-mean:

- `phi_x(i,j)  = (-1)^i`
- `phi_y(i,j)  = (-1)^j`
- `phi_xy(i,j) = (-1)^(i+j)`

Mọi thành phần tuần hoàn `2 x 2` không-DC đều có thể biểu diễn trong span của ba hàm này:

`R = a_x phi_x + a_y phi_y + a_xy phi_xy + W`

trong đó:

- `a_x, a_y, a_xy` là biên độ tuần hoàn,
- `W` là nhiễu zero-mean còn lại.

Định nghĩa matched-filter energy:

`T_k(R) = <R, phi_k>^2 / (N ||R||_2^2)`, với `k in {x, y, xy}`.

Trong đó:

- `N` là số pixel của residual,
- `<.,.>` là tích vô hướng Euclid.

### 4.1.3. Diễn giải

- Nếu không có thành phần tuần hoàn `2 x 2`, kỳ vọng của `T_k` là nhỏ.
- Nếu có periodicity ổn định kiểu CFA, `T_k` tăng bền vững.

Điều quan trọng là:

- đây là statistic có định hướng vật lý rõ,
- nó khác bản chất với các summary compression-heavy trong baseline.

### 4.1.4. Feature đã thử

Khối CFA gồm 6 feature:

- `cfa_cr_pi_x`
- `cfa_cr_pi_y`
- `cfa_cr_pi_xy`
- `cfa_cb_pi_x`
- `cfa_cb_pi_y`
- `cfa_cb_pi_xy`

## 4.2. Họ B: noise-level function

### 4.2.1. Động cơ vật lý

Trong camera, phương sai nhiễu thường phụ thuộc vào mức tín hiệu. Dưới mô hình đơn giản:

`Var[Z | mu] ~= a mu + b`

trong đó:

- `mu` là mức sáng kỳ vọng,
- `a > 0`,
- `b >= 0`.

Sau ISP, tone mapping, denoising và JPEG web, quan hệ này bị biến dạng nhưng đôi khi vẫn còn dấu vết ở các block phẳng.

### 4.2.2. Statistic

Trên các block phẳng `8 x 8` của `Y`, báo cáo thử 5 feature:

- `nlf_spearman`
- `nlf_slope`
- `nlf_intercept`
- `nlf_r2`
- `nlf_monotone_violation`

### 4.2.3. Giới hạn đã biết

Đây là họ feature đúng về mặt vật lý, nhưng có nhược điểm lớn:

- ảnh web thường đã qua ISP, resize, JPEG và khử nhiễu,
- vì vậy luật nhiễu gốc của sensor bị bào mòn mạnh.

Ta kỳ vọng họ này hữu ích hơn trên dữ liệu gần-raw, và yếu hơn trên dữ liệu web như repo hiện tại.

## 4.3. Họ C: phụ thuộc đa tỉ lệ kiểu wavelet

### 4.3.1. Động cơ thống kê

Ảnh tự nhiên không chỉ có histogram wavelet đuôi nặng; chúng còn có phụ thuộc xuyên mức giữa hệ số cha và hệ số con.

Viết một mô hình đơn giản:

`W = sqrt(Z) U`

trong đó:

- `U` là Gaussian nền,
- `Z >= 0` là scale ẩn.

Khi `Z` được chia sẻ cục bộ, độ lớn của coefficient ở hai mức wavelet sẽ có tương quan dương.

### 4.3.2. Feature đã thử

Khối wavelet gồm 5 feature:

- `wav_parent_corr_h`
- `wav_parent_corr_v`
- `wav_parent_corr_d`
- `wav_kurtosis_l1`
- `wav_kurtosis_l2`

### 4.3.3. Kỳ vọng

Nếu baseline hiện tại chưa nắm bắt tốt phụ thuộc đa tỉ lệ, wavelet sẽ bổ sung giá trị mới.  
Nếu baseline đã mã hóa gần tương đương qua FFT / residual / spatial summaries, gain của wavelet sẽ nhỏ.

## 5. Protocol thực nghiệm

## 5.1. Benchmark sạch

Để tránh geometry confound, báo cáo dùng lại protocol sạch:

- generator: `Midjourney`, `SDv14`, `SDv15`, `Wukong`
- `100 real + 100 fake` cho mỗi generator
- tổng `800` ảnh
- loại toàn bộ ảnh cần pad

## 5.2. Hai biến thể preprocessing dùng để đánh giá

### A. `pil_only_allshift`

- shift `+3` cố định cho mọi ảnh,
- JPEG bottleneck đơn, đối xứng và bảo thủ hơn.

Đây là benchmark bảo thủ, dùng làm căn cứ khuyến nghị chính.

### B. `method_a_allshift`

- shift `+3` cố định cho mọi ảnh,
- fake được precompress bất đối xứng kiểu nghiên cứu cũ,
- không dùng bottleneck cuối.

Biến thể này mạnh hơn nhưng không phải deploy default. Nó chỉ được dùng như upper-bound để xem họ feature nào hưởng lợi khi dấu vết image-formation còn nổi mạnh.

## 5.3. Classifier và metric

Mọi so sánh dùng cùng một classifier:

- median imputation,
- z-score,
- logistic regression với balanced class weights.

Metric chính:

- `logo_auc`

Metric phụ:

- `stratified_auc`

## 6. Kết quả chính

## 6.1. Benchmark bảo thủ `pil_only_allshift`

| Tập feature | Số chiều | `logo_auc` | Delta so với baseline |
|---|---:|---:|---:|
| `baseline33` | `33` | `0.7377` | `0.0000` |
| `baseline + CFA` | `39` | `0.7535` | `+0.0158` |
| `baseline + NLF` | `38` | `0.7401` | `+0.0025` |
| `baseline + wavelet` | `38` | `0.7418` | `+0.0041` |
| `baseline + all new` | `49` | `0.7548` | `+0.0171` |

### Diễn giải

- Khối CFA gần như thu hồi toàn bộ gain của toàn bộ feature mới.
- Tỉ lệ gain do CFA thu hồi là khoảng `92%` (`0.0158 / 0.0171`).
- NLF và wavelet đều có ích rất nhỏ trên benchmark bảo thủ.

Đây là bằng chứng mạnh nhất trong báo cáo.

## 6.2. Benchmark upper-bound `method_a_allshift`

| Tập feature | Số chiều | `logo_auc` | Delta so với baseline |
|---|---:|---:|---:|
| `baseline33` | `33` | `0.7594` | `0.0000` |
| `baseline + CFA` | `39` | `0.8492` | `+0.0898` |
| `baseline + NLF` | `38` | `0.7594` | `-0.0000` |
| `baseline + wavelet` | `38` | `0.7606` | `+0.0012` |
| `baseline + all new` | `49` | `0.8403` | `+0.0808` |

### Diễn giải

Khi preprocessing giữ mạnh dấu vết image-formation:

- CFA trở nên cực kỳ giàu thông tin,
- NLF vẫn gần như không giúp,
- wavelet giúp rất ít,
- toàn bộ gain vẫn bị dẫn dắt chủ yếu bởi CFA.

## 6.3. Hành vi cross-generator của `baseline + CFA`

Trên benchmark bảo thủ `pil_only_allshift`:

| Generator hold-out | Baseline | Baseline + CFA |
|---|---:|---:|
| `Midjourney` | `0.7962` | `0.8412` |
| `SDv14` | `0.7207` | `0.7286` |
| `SDv15` | `0.7730` | `0.7823` |
| `Wukong` | `0.6803` | `0.6990` |

Gain không bị khóa vào một generator duy nhất. Midjourney tăng mạnh nhất, nhưng cả bốn generator đều tăng.

## 6.4. Tính trực giao với baseline 33

| Họ feature | Mean max abs corr tới baseline | Max abs corr tới baseline |
|---|---:|---:|
| `CFA` | `0.16 - 0.17` | `0.20 - 0.22` |
| `NLF` | `0.22 - 0.23` | `0.37 - 0.42` |
| `Wavelet` | `0.55 - 0.56` | `0.76` |

Đây là chìa khóa giải thích thứ hạng kết quả:

- CFA giúp nhiều vì nó trực giao nhất,
- wavelet chồng lấn khá mạnh với baseline,
- NLF khá trực giao nhưng tín hiệu quá yếu trên ảnh web.

## 6.5. Latency

| Khối feature | ms / ảnh |
|---|---:|
| `baseline33` | `46.75` |
| `CFA` | `6.46` |
| `NLF` | `4.40` |
| `Wavelet` | `2.15` |
| `All new` | `13.45` |
| `baseline + all new` | `60.20` |

Diễn giải:

- thêm CFA làm tăng khoảng `13.8%` latency so với baseline,
- thêm toàn bộ họ mới làm tăng khoảng `28.8%`,
- xét theo gain / latency, CFA thắng với khoảng cách lớn.

## 7. Kết luận khoa học

## 7.1. Điều có thể khẳng định chắc

1. Baseline 33 thiếu một trục vật lý quan trọng: dấu vết CFA / demosaicing.
2. Trong ba họ được thử, CFA là họ duy nhất cho gain đáng kể và ổn định.
3. Wavelet có thông tin, nhưng phần lớn trùng lặp với baseline hiện có.
4. NLF là ý tưởng vật lý hợp lý nhưng quá yếu trên ảnh web của repo này.

## 7.2. Điều không nên suy diễn quá mức

1. Báo cáo này không chứng minh CFA luôn thắng trên mọi dataset.
2. Các con số trên `method_a_allshift` không nên dùng làm kỳ vọng deploy.
3. Gain của CFA phụ thuộc vào việc preprocessing có còn giữ được dấu vết image-formation hay không.

## 8. Khuyến nghị triển khai

## 8.1. Ưu tiên 1: thêm ngay khối CFA 6 feature

Thứ tự triển khai hợp lý nhất:

1. thêm 6 feature CFA,
2. retrain trên pipeline geometry-safe mới,
3. audit lại nuisance predictability,
4. chạy ablation đồng thời giữa `dct_mid_*` và khối CFA.

Đây là bước có xác suất thành công cao nhất.

## 8.2. Ưu tiên 2: chỉ thêm wavelet rất chọn lọc nếu cần headroom

Nếu sau khi thêm CFA vẫn cần cải thiện thêm, chỉ nên cân nhắc:

- `wav_parent_corr_h`
- `wav_parent_corr_v`
- tùy chọn `wav_parent_corr_d`

Không nên mặc định thêm cả khối wavelet kurtosis.

## 8.3. Ưu tiên 3: giữ NLF ở mức nghiên cứu

NLF nên ở trạng thái:

- chưa ship mặc định,
- chỉ dùng nếu sau này repo có dữ liệu gần-raw hơn hoặc pipeline mới bảo toàn sensor-noise tốt hơn.

## 8.4. Những gì không khuyến nghị

Không khuyến nghị mở rộng tiếp theo bằng:

- thêm nhiều feature lưới JPEG / DCT histogram,
- PRNU / sensor fingerprint trên ảnh web nén nhiều,
- các feature phase bậc cao đắt nhưng chưa được kiểm chứng trong repo này.

## 9. Kết luận

Kết luận thực dụng nhất cho repo là:

1. cleanup preprocessing trước,
2. sau đó thêm khối CFA trước tiên,
3. dùng LOGO AUC làm metric chính,
4. chỉ mở rộng thêm wavelet nếu cần,
5. không đặt kỳ vọng quá cao vào NLF trên dữ liệu web hiện tại.

Nếu chỉ được chọn **một** hướng mở rộng feature cho giai đoạn tiếp theo, hướng đó phải là:

**CFA periodicity**.
