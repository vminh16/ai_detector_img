# Reports Map

`docs/reports/` chua cac bao cao phan tich, validation, va tong hop bang chung.

## Current reports

- `shortcut_risk_validation.md`
  - bao cao hop nhat ve shortcut, compression history, geometry confound, chroma canonicalization, va cac phan quyet preprocessing

- `feature_space_update.md`
  - bao cao tong hop ve gioi han cua baseline 33-feature stack va huong mo rong feature space
  - ket luan thuc dung hien tai: uu tien CFA periodicity, wavelet co chon loc, NLF chi o muc nghien cuu

- `feature_spec_v2_validation.md`
  - validation moi nhat cho spec feature
  - bo sung study tren `Y-SRM` masking, local heteroskedasticity, edge consistency, resampling periodicity
  - bo sung diagnostics rieng cho SLA mapping, shift redundancy, control generalization, va cross-noise evaluation pathology
  - la bang chung chinh de khoa `feature_extraction_standard_v2.md`

## Historical / supporting reports

- `feature_spec_v1_validation.md`
  - validation truoc do tren `old_v1` vs `v4_exact`
  - giu lai de doi chieu su thay doi trong taxonomy

- `feature_spec_v0_review.md`
  - review trung gian cho `spec_feature_extract_v0`
  - giu lai de tham khao patch aggregation va cac huong mo rong som

## Nguyen tac

- `reports/` tra loi cau hoi "vi sao"
- `specs/` tra loi cau hoi "phai lam gi"
- Artifact so lieu, CSV, parquet, JSON metric khong de trong `reports/`; chung nam o `audit_output/`
