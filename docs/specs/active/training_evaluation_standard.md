# Đặc tả huấn luyện và đánh giá sau full notebook train

## 1. Trạng thái tài liệu

Đây là đặc tả active cho pha huấn luyện và đánh giá của nhánh `codex/preprocessing-v4-core` sau khi đã chạy xong notebook training/evaluation đầy đủ.

Tài liệu này không chỉ mô tả workflow, mà còn khóa các kết luận đã được xác nhận bằng artifact của full run ngày `2026-04-03`.

Nguồn bằng chứng chuẩn của tài liệu này:

- [summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/summary.json)
- [candidate_val_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/candidate_val_metrics.csv)
- [selected_model_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_metrics.csv)
- [selected_model_ood_by_generator.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_ood_by_generator.csv)
- [selected_model_family_importance.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_family_importance.csv)
- [clean_cfa_gate_coverage.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/clean_cfa_gate_coverage.csv)
- [model_level_auc_nat.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase2_model_nuisance/model_level_auc_nat.csv)
- [nuisance_label_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase2_model_nuisance/nuisance_label_summary.csv)
- [degradation_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase3_degradation_suite/degradation_metrics.csv)
- [degradation_gap_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase3_degradation_suite/degradation_gap_summary.csv)
- [degradation_cfa_gate_coverage.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase3_degradation_suite/degradation_cfa_gate_coverage.csv)
- [ablation_candidate_val_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase4_family_ablation/ablation_candidate_val_metrics.csv)
- [ablation_clean_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase4_family_ablation/ablation_clean_metrics.csv)
- [branch_closure_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase4_family_ablation/branch_closure_summary.csv)
- [phase_closure_summary.json](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase5_phase_closure/phase_closure_summary.json)

Notebook và orchestration tương ứng:

- [03_training_eval.ipynb](C:/Users/USER/Desktop/ai_detector_img/notebooks/03_training_eval.ipynb)
- [run_training_phase_closure](C:/Users/USER/Desktop/ai_detector_img/src/training/phase_closure.py)

## 2. Vai trò thật sự của pha này

Pha huấn luyện và đánh giá của nhánh hiện tại phải trả lời bốn câu hỏi:

1. với feature table `v2`, tổ hợp `feature branch × model family` nào đang mạnh nhất trên benchmark clean
2. mô hình đó có đang học `subsampling / codec history` ở mức hệ thống hay không
3. mô hình đó có giữ được utility khi patch canonical đi qua các degradation phổ biến ngoài đời hay không
4. branch nào còn đáng giữ cho vòng tiếp theo, branch nào chỉ nên giữ ở mức research/reference

Pha này không có mục tiêu:

- chốt champion deploy
- chốt threshold deploy cuối
- hay tuyên bố rằng bài toán shortcut đã được giải xong

Kết luận của pha này chỉ là kết luận khoa học của vòng benchmark sau full run.

## 3. Input contract đã được khóa

Đầu vào duy nhất của pha này là feature table:

- [feature_extraction_v2_rgb248_exact.csv](C:/Users/USER/Desktop/ai_detector_img/features/feature_extraction_v2_rgb248_exact.csv)

Full run xác nhận:

- `rows = 85,615`
- `feature_version = v2_rgb248_exact_multibranch`
- `preprocess_version = v4_rgb248_r4_exact`
- `required_audits_completed = true`

Ba điều kiện cứng vẫn giữ nguyên:

1. chỉ dùng feature table có `feature_version = v2_rgb248_exact_multibranch`
2. chỉ dùng feature table có `preprocess_version = v4_rgb248_r4_exact`
3. chỉ dùng các hàng có `status = ok`

Không được thay source-of-truth bằng:

- metadata stale cũ
- cờ mode/format cũ
- hay bất kỳ bảng ngoài nào làm thay đổi nhãn hoặc split contract

## 4. Ký hiệu và metric chuẩn

### 4.1. Ký hiệu cơ bản

- `x_i`: vector đặc trưng của hàng `i`
- `y_i ∈ {0,1}`
  - `1`: ảnh AI
  - `0`: ảnh real
- `f_theta`: mô hình nền fit trên `train_core`
- `g_phi`: Platt calibrator fit trên `calibration`
- `p_i = g_phi(f_theta(x_i))`: xác suất sau calibration
- `tau`: threshold khóa trên `val`

### 4.2. Split contract

Feature table active phải có đủ năm split:

- `train_core`
- `calibration`
- `val`
- `id_test`
- `ood_eval`

Ý nghĩa:

- `train_core`: fit mô hình nền
- `calibration`: fit Platt scaling
- `val`: chọn candidate và khóa threshold
- `id_test`: holdout in-domain
- `ood_eval`: holdout out-of-domain

### 4.3. Clean metric

Với mỗi candidate, phải báo:

- `AUC`
- `Brier`
- `ECE`
- `TPR`
- `FPR`
- `precision`
- `accuracy`
- `threshold`

Threshold được khóa theo quy tắc:

`tau = min { t : FPR_val(t) <= 5% }`

### 4.4. Nuisance metric ở mức model

`AUC_nat_raw` là ROC AUC khi dùng chính score của model để tách:

- real JPEG có subsampling gốc `4:2:0`
- real JPEG có subsampling gốc `4:4:4`

Trong tài liệu này dùng:

`AUC_nat_abs = max(AUC_nat_raw, 1 - AUC_nat_raw)`

Ý nghĩa:

- `AUC_nat_abs = 0.5`: model không tách được nuisance real-only
- `AUC_nat_abs` càng cao: model càng mang thông tin về natural subsampling history

### 4.5. Cross-degradation metric

Với degradation `g`:

`AUC_xdeg(g) = AUC(score(model_train_clean, Phi(g(X_can)))), Y)`

Trong đó:

- `X_can`: patch canonical `248x248`
- `g`: degradation tác động trực tiếp trên patch canonical
- `Phi(.)`: pipeline feature extraction active

Các gap phải báo:

- `auc_gap = AUC_xdeg(g) - AUC_clean`
- `brier_gap = Brier_xdeg(g) - Brier_clean`
- `ece_gap = ECE_xdeg(g) - ECE_clean`
- `tpr_gap = TPR_xdeg(g) - TPR_clean`
- `fpr_gap = FPR_xdeg(g) - FPR_clean`

### 4.6. CFA gate coverage

Với branch có CFA điều kiện, phải báo:

`gate_rate = mean(1[cfa_validity_score >= tau_cfa])`

Coverage này luôn phải được báo theo:

- split
- label
- degradation

vì CFA không phải feature always-on.

## 5. Workflow chuẩn đã được xác nhận

Full notebook train đã khép pha theo đúng năm bước:

1. clean benchmark
2. model-level `AUC_nat`
3. degradation suite
4. family ablation
5. phase closure summary

Các pha này tương ứng với các artifact con:

- `phase1_clean_benchmark`
- `phase2_model_nuisance`
- `phase3_degradation_suite`
- `phase4_family_ablation`
- `phase5_phase_closure`

Đây là workflow chuẩn bắt buộc cho mọi vòng benchmark tiếp theo của nhánh.

## 6. Kết quả pha 1: clean benchmark

### 6.1. Candidate mạnh nhất trên validation

Full run chọn:

- `selected_clean_candidate = full_v2__lightgbm`
- `selected_clean_feature_set = full_v2`
- `selected_clean_model_name = lightgbm`
- `selected_clean_val_auc = 0.9548272480`
- `selected_clean_threshold = 0.7074744499`

Top clean candidates theo [candidate_val_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/candidate_val_metrics.csv):

1. `full_v2__lightgbm`: `val_auc = 0.954827`
2. `always_on_plus_cfa_raw__lightgbm`: `0.933964`
3. `full_v2__logreg`: `0.914035`
4. `always_on_plus_cfa_raw__logreg`: `0.904559`
5. `always_on_plus_cfa_gated__lightgbm`: `0.868869`

### 6.2. Metrics clean của selected candidate

Theo [selected_model_metrics.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_metrics.csv):

- `val`: `AUC = 0.954827`, `TPR = 0.795333`, `FPR = 0.049982`, `Brier = 0.083164`, `ECE = 0.014494`, `AUC 95% CI = [0.950817, 0.959515]`
- `id_test`: `AUC = 0.949068`, `TPR = 0.774667`, `FPR = 0.051755`, `Brier = 0.089715`, `ECE = 0.010927`, `CI = [0.944785, 0.954154]`
- `ood_eval`: `AUC = 0.967559`, `TPR = 0.851701`, `FPR = 0.050231`, `Brier = 0.070428`, `ECE = 0.024139`, `CI = [0.965858, 0.969356]`
- `pooled_eval`: `AUC = 0.962871`, `TPR = 0.831683`, `FPR = 0.050420`, `Brier = 0.075201`, `ECE = 0.016546`, `CI = [0.961181, 0.964462]`

### 6.3. Breakdown theo generator

Theo [selected_model_ood_by_generator.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_ood_by_generator.csv):

- `GLIDE`: `AUC = 0.981562`
- `SDv15`: `AUC = 0.956878`

Kết luận pha clean:

- clean benchmark hiện rất mạnh
- `lightgbm` vượt rõ `logreg`
- `full_v2` là feature-set có utility clean mạnh nhất của vòng benchmark này

Nhưng kết luận clean một mình chưa đủ để chốt model của nhánh.

## 7. Kết quả pha 2: model-level nuisance audit

### 7.1. Nền nhãn nuisance thực tế

Theo [nuisance_label_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase2_model_nuisance/nuisance_label_summary.csv), tập real-only có đủ support `4:2:0` và `4:4:4` ở mọi split đánh giá:

- `val`: `265` mẫu `4:2:0`, `2527` mẫu `4:4:4`
- `id_test`: `301` và `2492`
- `ood_eval`: `1143` và `12160`
- `pooled_eval`: tổng hợp của ba split trên

### 7.2. `AUC_nat_abs` của các branch chính

Theo [model_level_auc_nat.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase2_model_nuisance/model_level_auc_nat.csv), `pooled_eval` cho các branch chính là:

- `always_on__lightgbm`: `0.756451`
- `always_on_plus_cfa_gated__lightgbm`: `0.720761`
- `always_on_plus_wavelet__lightgbm`: `0.752964`
- `always_on_plus_ysrm__lightgbm`: `0.643436`
- `full_v2__lightgbm`: `0.671035`
- `full_v2_minus_conditional_cfa__lightgbm`: `0.668082`
- `full_v2_minus_wavelet_decay__lightgbm`: `0.664802`
- `full_v2_minus_dark_textured_hetero__lightgbm`: `0.663772`
- `full_v2_minus_content_adaptive_y_srm__lightgbm`: `0.702737`

### 7.3. Diễn giải chuẩn

Các số liệu trên khóa ba kết luận:

1. không có branch nào đưa `AUC_nat_abs` về gần `0.5`
2. `always_on` không hề là branch “sạch ở mức model”; ngược lại nó tách nuisance mạnh hơn `full_v2`
3. `full_v2__lightgbm` giảm được nuisance so với `always_on`, nhưng vẫn còn `AUC_nat_abs = 0.671035`, nên chưa thể coi là robust trước natural subsampling history

Vì vậy:

- clean utility cao không đủ để kết luận model đang học khác biệt thật giữa ảnh diffusion và ảnh real
- `AUC_nat` phải luôn được đọc cùng với clean AUC

## 8. Kết quả pha 3: degradation suite

### 8.1. Suite active

Full run đã đánh giá đủ năm degradation:

- `jpeg95_420`
- `jpeg90_420`
- `resize75_bilinear`
- `resize50_bilinear`
- `resize50_jpeg90_420`

### 8.2. `full_v2__lightgbm` dưới degradation

Theo [degradation_gap_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase3_degradation_suite/degradation_gap_summary.csv), trên `pooled_eval`:

- clean: `AUC = 0.962871`, `TPR = 0.831683`, `FPR = 0.050420`
- `jpeg95_420`: `AUC = 0.855707`, `auc_gap = -0.107165`, `TPR = 0.366547`, `FPR = 0.045540`
- `jpeg90_420`: `AUC = 0.836182`, `auc_gap = -0.126690`, `TPR = 0.309724`, `FPR = 0.044806`
- `resize75_bilinear`: `AUC = 0.762295`, `auc_gap = -0.200576`, `TPR = 0.754102`, `FPR = 0.401626`
- `resize50_bilinear`: `AUC = 0.597578`, `auc_gap = -0.365293`, `TPR = 0.943327`, `FPR = 0.904669`
- `resize50_jpeg90_420`: `AUC = 0.588691`, `auc_gap = -0.374180`, `TPR = 0.632353`, `FPR = 0.505509`

### 8.3. Kết luận vật lý từ degradation suite

Kết quả trên khóa ba điểm:

1. JPEG-only degradation làm giảm mạnh utility, nhưng chưa phải failure mode tệ nhất
2. resize là failure mode chính của branch tốt nhất hiện tại
3. dưới resize, threshold khóa từ clean không còn giữ được hành vi phân loại có ý nghĩa; FPR tăng đột biến lên mức không chấp nhận được

Nói cách khác:

- model hiện tại chưa ổn trước hậu kỳ scale-change
- clean `AUC` cao không chuyển dịch thành robustness ngoài đời

### 8.4. CFA gate coverage dưới degradation

Theo [clean_cfa_gate_coverage.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/clean_cfa_gate_coverage.csv), trên clean:

- `train_core`: AI `23.6%`, real `26.5%`
- `val`: AI `23.6%`, real `26.9%`
- `id_test`: AI `23.1%`, real `25.9%`
- `ood_eval`: AI `8.7%`, real `26.9%`

Theo [degradation_cfa_gate_coverage.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase3_degradation_suite/degradation_cfa_gate_coverage.csv):

- `jpeg95_420`: gate gần như tắt hoàn toàn, thường chỉ `0.1%` đến `0.5%`
- `jpeg90_420`: gate vẫn gần như tắt, thường dưới `1%`
- `resize75_bilinear`: AI giữ được rất ít (`~1.8%` đến `2.7%`), real gần như bằng `0`
- `resize50_bilinear`: gần như `0` ở mọi split
- `resize50_jpeg90_420`: vẫn gần `0`

Kết luận:

- `conditional CFA` là branch điều kiện đúng nghĩa
- branch này không được diễn giải như feature always-on
- coverage collapse dưới JPEG/resize là bằng chứng thực nghiệm rằng CFA phải được giữ như cue điều kiện, không phải cue phổ quát

## 9. Kết quả pha 4: family ablation

### 9.1. Tóm tắt branch-level

Theo [branch_closure_summary.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase4_family_ablation/branch_closure_summary.csv):

- `full_v2__lightgbm`
  - `clean_pooled_auc = 0.962871`
  - `auc_nat_abs = 0.671035`
  - `worst_xdeg_auc = 0.588691`
  - `mean_xdeg_auc = 0.728091`

- `full_v2_minus_wavelet_decay__lightgbm`
  - `clean_pooled_auc = 0.959992`
  - `auc_nat_abs = 0.664802`
  - `worst_xdeg_auc = 0.597687`
  - `mean_xdeg_auc = 0.727981`

- `full_v2_minus_dark_textured_hetero__lightgbm`
  - `clean_pooled_auc = 0.959833`
  - `auc_nat_abs = 0.663772`
  - `worst_xdeg_auc = 0.579928`
  - `mean_xdeg_auc = 0.720186`

- `full_v2_minus_content_adaptive_y_srm__lightgbm`
  - `clean_pooled_auc = 0.950171`
  - `auc_nat_abs = 0.702737`
  - `worst_xdeg_auc = 0.613867`
  - `mean_xdeg_auc = 0.740164`

- `full_v2_minus_conditional_cfa__lightgbm`
  - `clean_pooled_auc = 0.927908`
  - `auc_nat_abs = 0.668082`
  - `worst_xdeg_auc = 0.586208`
  - `mean_xdeg_auc = 0.738780`

- `always_on__lightgbm`
  - `clean_pooled_auc = 0.878468`
  - `auc_nat_abs = 0.756451`
  - `worst_xdeg_auc = 0.624077`
  - `mean_xdeg_auc = 0.747007`

### 9.2. Ý nghĩa của từng family

Từ ablation summary và [selected_model_family_importance.csv](C:/Users/USER/Desktop/ai_detector_img/audit_output/validation/training_v2_phase_closure_20260403/phase1_clean_benchmark/selected_model_family_importance.csv), thứ tự ảnh hưởng của các family trong selected model là:

1. `content_adaptive_y_srm`
2. `conditional_cfa`
3. `fft_midband`
4. `control_spatial`
5. `wavelet_decay`
6. `control_color`
7. `dark_textured_hetero`
8. `control_frequency`

Các kết luận được khóa:

- bỏ `conditional_cfa` làm clean AUC giảm mạnh nhất; branch này mang utility clean rất lớn
- bỏ `content_adaptive_y_srm` cũng làm clean AUC giảm rõ; branch này mang utility quan trọng
- bỏ `wavelet_decay` chỉ làm clean AUC giảm rất ít; wavelet hiện là family phụ, chưa phải trụ cột
- bỏ `dark_textured_hetero` gần như không đổi clean AUC; family này hiện có utility bổ sung nhỏ

Nhưng family importance không được phép suy diễn thành “an toàn”:

- `conditional_cfa` là branch utility cao nhưng coverage điều kiện hẹp
- `content_adaptive_y_srm` là branch utility cao nhưng vẫn nằm trong nhóm residual microtexture nhạy hậu kỳ
- `wavelet_decay` đóng góp nhỏ, nên nên giữ ở vai trò phụ hoặc research

### 9.3. Không có branch nào thắng toàn diện

Từ branch summary:

- `full_v2__lightgbm` thắng clean nhưng thua rõ về robustness với resize
- `always_on__lightgbm` chịu degradation đỡ hơn ở `worst_xdeg_auc`, nhưng clean yếu hơn nhiều và `AUC_nat` còn tệ hơn
- `always_on_plus_ysrm__lightgbm` có `AUC_nat_abs` thấp nhất (`0.643436`) nhưng clean chỉ `0.907612` và `worst_xdeg_auc` vẫn chỉ `0.587858`
- `always_on_plus_cfa_gated__lightgbm` có `worst_xdeg_auc` tốt hơn (`0.649657`), nhưng clean chỉ `0.899724` và `AUC_nat_abs` vẫn cao (`0.720761`)

Kết luận cứng:

> Sau full run, không có branch nào đồng thời tối ưu cả ba trục:
> `clean utility`, `natural nuisance`, và `cross-degradation robustness`.

## 10. Phán quyết chuẩn của pha này

### 10.1. Model tham chiếu của vòng benchmark

Model tham chiếu của vòng benchmark hiện tại là:

- `full_v2__lightgbm`

Lý do:

- đứng đầu validation
- đứng đầu clean pooled evaluation
- tốt nhất trên clean among all audited branches

### 10.2. Nhưng model tham chiếu chưa phải champion

`full_v2__lightgbm` chưa đạt chuẩn champion của nhánh vì:

1. `AUC_nat_abs = 0.671035` vẫn còn đủ cao để kết luận model còn mang thông tin về natural subsampling history
2. `worst_xdeg_auc = 0.588691` cho thấy failure mode nghiêm trọng dưới resize-heavy degradation
3. family importance cho thấy model đang dựa nhiều vào:
   - `content_adaptive_y_srm`
   - `conditional_cfa`
   - `wavelet_decay`

đều là các branch chưa được coi là universally safe

### 10.3. Phán quyết theo branch

- `always_on`
  - không còn được phép gọi là “safe branch”
  - clean yếu hơn rõ rệt
  - `AUC_nat` còn cao hơn `full_v2`

- `conditional CFA`
  - utility clean rất quan trọng
  - nhưng chỉ hợp lệ như branch điều kiện
  - coverage collapse dưới JPEG/resize đã được chứng minh bằng dữ liệu

- `content_adaptive_y_srm`
  - utility clean cao
  - nhưng vẫn là branch rủi ro, chưa đủ bằng chứng để nâng lên always-on core

- `wavelet_decay`
  - utility bổ sung hiện tại nhỏ
  - không nên là branch quyết định

- `dark_textured_hetero`
  - hiện chưa chứng minh được vai trò lớn ở mức model
  - giữ ở vai trò hỗ trợ/research

## 11. Logic chuẩn để diễn giải kết quả của nhánh

Pha training/evaluation sau full notebook train cho phép kết luận một cách chặt chẽ như sau:

1. preprocessing `v4_exact` và feature extraction `v2` đã đủ ổn định để tạo ra benchmark đáng tin ở mức hệ thống
2. feature branches hiện tại có thể tạo ra model clean rất mạnh
3. nhưng sức mạnh clean này chưa đồng nghĩa với việc model học được hoàn toàn khác biệt bản chất giữa ảnh diffusion và ảnh real
4. bằng chứng phản biện là:
   - `AUC_nat` vẫn cao
   - `resize-heavy degradation` làm model sụp mạnh
   - branch thắng clean đang dựa đáng kể vào các family chưa universally safe

Do đó, pipeline hiện tại:

- đã đủ để đóng pha benchmark
- chưa đủ để khóa model production/champion

## 12. Quy tắc bắt buộc cho vòng tiếp theo

Từ vòng benchmark này, mọi vòng sau phải giữ nguyên các nguyên tắc:

1. mọi candidate mới đều phải được so với `full_v2__lightgbm` trên cùng ba trục:
   - clean pooled AUC
   - `AUC_nat_abs`
   - `worst_xdeg_auc`

2. không được dùng clean AUC một mình để nâng branch/model lên champion

3. `conditional CFA` chỉ được dùng như branch điều kiện; không được hợp thức hóa thành feature always-on

4. `models/param/training_v2_phase_closure_20260403` chỉ là benchmark reference, không phải artifact deploy cuối

5. mọi vòng benchmark mới vẫn phải ghi đủ năm phase artifact như vòng hiện tại

6. nếu một candidate mới không cải thiện ít nhất một trong ba trục chính mà không làm trục còn lại sụp đổ, candidate đó không có giá trị chiến lược

## 13. Kết luận

Sau full notebook train, nhánh hiện tại đã có một kết luận khoa học đủ chặt:

- `full_v2__lightgbm` là model benchmark mạnh nhất hiện tại
- không có branch nào đủ sạch và đủ bền để được gọi là champion-ready
- failure mode chính của hệ thống hiện nay là `resize-heavy degradation`
- `AUC_nat` chứng minh rằng natural subsampling history vẫn chưa bị triệt khỏi hành vi ở mức model
- `conditional CFA` và `content_adaptive_y_srm` là hai nguồn utility lớn nhất hiện tại, nhưng cũng là hai nơi cần được diễn giải và kiểm soát chặt nhất

Tài liệu này vì vậy là đặc tả hậu-training của nhánh:

- đủ để kết thúc pha training/evaluation hiện tại
- đủ để khóa reference benchmark
- và đủ để đặt chuẩn cho vòng nghiên cứu/thiết kế branch tiếp theo
