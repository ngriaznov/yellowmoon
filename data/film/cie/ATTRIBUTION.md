# CIE tables

```
CIE colour-science tables, from the International Commission on Illumination (CIE), Vienna.
Licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ (the text is in LICENSE, in this folder).
Not modified: the CSV and metadata files are byte for byte as published.
```

The Film node reads these files at run time (the observer for stages 1 and 9, D50 and D55 as lights). They are plain files,
never compiled into the program. A table resampled or integrated in memory is not written anywhere.

Fetched: 2026-11-04, from https://files.cie.co.at/Publications-datasets/ (the file links on each dataset page).

Licence as the source states it: each dataset page on cie.co.at links a metadata file (`<name>.csv_metadata.json`) whose
`rightsList` says "Creative Commons Attribution-ShareAlike 4.0 International" (`CC BY-SA 4.0`, https://creativecommons.org/licenses/by-sa/4.0/).
The text of the dataset pages themselves names no licence. The metadata files are kept here as published.

| File | Table (the metadata file's description) | Dataset page | Cite as |
|---|---|---|---|
| `CIE_xyz_1931_2deg.csv` | CIE 1931 colour-matching functions (x_bar, y_bar, z_bar), 2 degree observer, wavelength range: 360 nm to 830 nm, wavelength increment: 1 nm, original source: CIE 018:2019, Table 6. | https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer | CIE 2019, Colour-matching functions of CIE 1931 standard colorimetric observer, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.xvudnb9b |
| `CIE_std_illum_D50.csv` | CIE standard illuminant D50, wavelength range: 300 nm to 830 nm, wavelength increment: 1 nm, original source: ISO/CIE 11664-2:2022, Table B.1 | https://cie.co.at/datatable/cie-standard-illuminant-d50 | CIE 2022, CIE standard illuminant D50, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.etgmuqt5 |
| `CIE_illum_D55.csv` | Relative spectral power distributions of CIE illuminant D55, wavelength range: 380 nm to 780 nm, wavelength increment: 5 nm, original source: CIE 015:2018 table 5 | https://cie.co.at/datatable/relative-spectral-power-distributions-cie-illuminant-d55 | CIE 2018, Relative spectral power distributions of CIE illuminant D55, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.qewfb3kp |

Modified by: nobody. If a table is ever modified, the change is described in `CHANGELOG.txt` beside this file, with who made it.
Written by `tools/film/fetch-cie.sh`; do not edit by hand.
