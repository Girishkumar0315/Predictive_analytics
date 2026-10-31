# Dataset documentation — Stack Overflow Annual Developer Survey 2024

## 1. What the data is
Responses of **65,437 developers** to the 2024 Stack Overflow Annual Developer Survey (fielded May 2024).
Topics: demographics, education, work and career, technology culture, Stack Overflow community use, and **artificial
intelligence (AI) tools in the development workflow**. It is real, self-reported survey data — nothing in this project
was simulated or fabricated.

## 2. Official provenance (please verify yourself)
| Item | Value |
|---|---|
| Original publisher | Stack Overflow (Prosus Media) |
| Official survey / data page | https://survey.stackoverflow.co/ (2024 results: https://survey.stackoverflow.co/2024/) |
| Original licence | Open Database Licence (ODbL) as stated by Stack Overflow on its survey page — **please re-confirm on the page before submission** |
| Curated extract used here | TidyTuesday 2024 week 36 "Stack Overflow Annual Developer Survey 2024", curated by Havisha Khurana |
| Extract location | https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-09-03 |
| Raw file URL | https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-09-03/stackoverflow_survey_single_response.csv |
| Repository commit at download time | `b2258ae949e3c64099c4e6d00ff4b68dba1d6d0e` (downloaded October 2026) |
| What the extract contains | Only the **single-response** questions of the main survey, **integer-coded**, plus a crosswalk (code → label) and the question text. Multi-select questions (e.g. languages, tools) are *not* included. |

### Honest disclosure about how the data was obtained
The sources named in the assignment brief (Kaggle, data.gov.in, Mendeley Data, AI Kosh) and Stack Overflow's own download
CDN were **not reachable from the environment in which this project was built**, so the file could not be downloaded
from them. The TidyTuesday GitHub repository *was* reachable and hosts a documented, openly licensed copy derived directly
from the official Stack Overflow release (its cleaning script is in `data/raw/TIDYTUESDAY_README_ORIGINAL.md`).
If your faculty requires the file to come from the original portal, download the official zip from
https://survey.stackoverflow.co/ — the notebook logic is identical; only the code→label mapping step (`decode()`)
would need adapting because the official file stores text labels instead of integer codes.

## 3. Files in `data/raw/` (byte-for-byte copies)
| File | Rows × cols | SHA-256 |
|---|---|---|
| `stackoverflow_survey_single_response.csv` | 65,437 × 28 | `bbfad6bf7ee6b2f511656ea15637a7c07265d70d6b9c785e0bb8c3009cec2446` |
| `qname_levels_single_response_crosswalk.csv` | 122 × 3 | `3d73865ee768bc932bf83ba7346e676fb08c5ceab3b0886162987d4c067f4327` |
| `stackoverflow_survey_questions.csv` | 24 × 2 | `0db3914e4b2e671e31e6d92c0aedea6be147b8456f254253883873ee93637933` |
| `meta.yaml`, `TIDYTUESDAY_README_ORIGINAL.md` | — | original metadata / readme / cleaning script |

Verify with: `sha256sum data/raw/*.csv`.

## 4. Data dictionary
See `reports/results/tables/tbl_data_dictionary.csv` (also Table 3.1 of the report). Key columns:
`converted_comp_yearly` (USD, **prediction target 1**), `ai_select` (uses AI tools, **prediction target 2**), `country`,
`dev_type`, `years_code_pro`, `org_size`, `remote_work`, `ed_level`, `age`, Stack Overflow engagement columns, and the four
AI follow-up questions (`ai_sent`, `ai_acc`, `ai_complex`, `ai_threat`).

## 5. Known limitations of the data
* **Self-selected sample** (people who visit/answer Stack Overflow surveys) — not a random sample of all developers.
* **Compensation is self-reported, optional, and nominal** (not purchasing-power adjusted).
* 64 % of respondents have no converted salary; 10,305 typed a salary whose currency could not be converted.
* AI follow-up questions were shown **conditionally** → structural missingness (see report §3 and §4.5).
* Extract excludes multi-select questions (languages, tools, AI tools used), limiting richer feature engineering.
