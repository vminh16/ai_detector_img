# Data Map

`data/` chua raw snapshot va cac dataset da qua xu ly.

## Current layout

- `raw/`
  - raw snapshot dang duoc audit / preprocess

- `raw_cleaned_v1/`
  - cleaned snapshot cu, giu lai de doi chieu lich su

- `processed/`
  - output preprocessing cu

- `processed_v4_rgb248_r4_exact/`
  - output preprocessing v4
  - exact crop `248 x 248`, residue `(4,4)`, RGB `.npy`

## Nguyen tac

- `raw/` va `processed*/` la data artifact, khong phai source code
- Notebook va script phai ghi ro snapshot / manifest nao duoc dung
- Khong dat code, log tam, hay cache Python trong `data/`
