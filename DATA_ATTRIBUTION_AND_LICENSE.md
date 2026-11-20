# Data attribution and licensing

## Dataset
**Marine Engine Fault Dataset**, Version 1.0.
A. BahooToroody, O. Bondarenko, M. M. Abaei, Y. Niki, E. Zio (2026). Zenodo. https://doi.org/10.5281/zenodo.19857425
Data descriptor (preprint): *Marine Engine Fault Dataset: Open-Access Data under Controlled Reference and Fault Scenario Conditions*, arXiv:2607.19444.
Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/

## Modifications in this project (required by CC BY 4.0)
`data/Marine_Engine_Cleaned_Causal_v1.zip` is a **cleaned derivative**, not the original archive:
- Temperature values above 800 °C or below −20 °C were set to missing.
- Gaps were forward-filled causally for up to 10 consecutive rows. Longer gaps remain missing.
- The symbol and unit header rows were removed.
- No scaling, feature selection or model fitting was applied.

SHA256 of the derivative: `ea142347a30f28c040003216a45cb74e6a5fc09094e6a12c9c95356f1facb85e`.
The original creators do not endorse this project or its results.

## Equipment
Matsui Iron Works MU323DGSC three-cylinder, turbocharged/intercooled marine diesel engine on a test bench. That is **one engine**; no claim of cross-engine generalization is made.

## Software
The project code is provided for research and education. The third-party packages listed in `requirements.txt` keep their own licences.
