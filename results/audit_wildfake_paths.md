# WildFake path-level audit

From `data\audit_sample\wildfake_csv` — 6 generator file lists, 323,235 paths, zero images downloaded.

## Container format by generator

| list      |      n | class   |   advanced |   jpg% |   png% | root                                     |
|:----------|-------:|:--------|-----------:|-------:|-------:|:-----------------------------------------|
| BigGAN    |  15540 | ai      |          0 |   64.4 |   35.6 | ./GAN_based/Typical/BigGAN               |
| dalle2    |  55638 | ai      |          0 |    0   |  100   | ./Diffusion_based/DALLE/Typical          |
| dalle3    |   8843 | ai      |          1 |  100   |    0   | ./Diffusion_based/DALLE/Advanced         |
| imagen    |  47435 | ai      |          0 |    0   |  100   | ./Diffusion_based/Imagen/backpack_Chanel |
| real_afhq |  31933 | real    |          0 |  100   |    0   | ./Real/afhq/afhq                         |
| real_coco | 163846 | real    |          0 |  100   |    0   | ./Real/coco/coco2017                     |

## Forbidden reference subset (§4.3.2)

| path prefix | found | expected | matches |
|---|---:|---:|:--:|
| `./Real/coco/coco2017/val2017/` | 4998 | 4998 | yes |
| `./Diffusion_based/DALLE/Advanced/` | 8843 | 8843 | yes |

Paths *locate* the forbidden subset; they must not be what enforces the exclusion. WildFake has renamed COCO's files to `img000000.jpg`, so the original IDs are gone and any path check dies the moment a directory is reorganised. The manifest builder content-hashes these files and blocks by hash.

