# Documentation Map

## Cau truc

- `specs/`
  - dac ta ky thuat dang co hieu luc hoac dang la ban nhap chinh thuc cho pha tiep theo
- `reports/`
  - bao cao validation, phan bien, va tong hop bang chung
- `reference/`
  - tai lieu legacy va tham khao lich su

## Specs nen doc truoc

- `specs/preprocessing_pipeline_standard_v4.md`
  - source-of-truth cho preprocessing core hien tai

- `specs/feature_extraction_standard_v2.md`
  - source-of-truth moi nhat cho feature extraction
  - chot inventory feature v2, taxonomy `always-on / conditional / research-only / drop`, va huong fusion phi tuyen

- `specs/feature_extraction_standard_v1.md`
  - spec active truoc do
  - giu lai de doi chieu lap luan va migration sang v2

## Reports quan trong

- `reports/shortcut_risk_validation.md`
  - bao cao shortcut, compression history, geometry confound, va preprocessing risk

- `reports/feature_space_update.md`
  - bao cao gioi han baseline 33 feature va huong mo rong feature space

- `reports/feature_spec_v2_validation.md`
  - validation moi nhat tren `old_v1` vs `v4_exact`
  - bo sung diagnostics ve SLA mapping, shift redundancy, control generalization, cross-noise pathology, va framing `multi-feature + conditional branches`

- `reports/training_v2_baseline_20260403.md`
  - benchmark training baseline dau tien tren full feature table v2
  - ket luan quan trong: clean baseline manh, nhung selected model van phu thuoc vao cac family chua du nuisance-audit

- `reports/feature_spec_v1_validation.md`
  - bao cao validation truoc do
  - giu lai de doi chieu lich su ra quyet dinh

- `reports/feature_spec_v0_review.md`
  - review trung gian truoc vong validation moi nhat
  - giu lai de doi chieu cac gia thuyet mo rong feature space va patch aggregation

## Luu y quan tri tai lieu

- `specs/` tra loi cau hoi "phai lam gi"
- `reports/` tra loi cau hoi "vi sao"
- `reference/` chi giu gia tri tham khao lich su
- `specs/preprocessing_pipeline_standard.md` duoc giu lai de doi chieu, nhung khong con la active spec cho champion path
