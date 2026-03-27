# Specs Map

`docs/specs/` chi chua cac dac ta dang co hieu luc hoac dang la ban nhap chinh thuc cho pha tiep theo.

## Active / current

- `preprocessing_pipeline_standard_v4.md`
  - core preprocessing hien tai dang co hieu luc
  - geometry-safe exact crop, mode contract, support gate, audit-driven governance

- `feature_extraction_standard_v2.md`
  - source-of-truth hien tai cho pha feature extraction
  - khoa taxonomy `always-on / conditional / research-only / drop`
  - chot inventory v2 gom `control_minimal`, `fft_midband_y`, `conditional_cfa_rgb`, va cac research branches

- `feature_extraction_standard_v1.md`
  - ban active truoc do, da bi supersede boi `feature_extraction_standard_v2.md`
  - giu lai de doi chieu framing va taxonomy

- `feature_extraction_standard_v0.md`
  - ban nhap da bi supersede
  - giu lai de doi chieu lap luan va lich su governance

## Historical / superseded in-place

- `preprocessing_pipeline_standard.md`
  - giu lai de doi chieu lap luan va lich su ra quyet dinh
  - khong con la source-of-truth cho champion preprocessing

## Nguyen tac

- Moi spec moi phai tham chieu artifact cu the trong `audit_output/`
- Neu mot spec bi supersede, cap nhat file nay truoc khi cap nhat README root
- Khong dat tai lieu report vao `specs/`
