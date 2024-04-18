## MARLP: Time-series Forecasting Control for Agricultural Managed Aquifer Recharge

🌊 Welcome to the MARLP repository, featuring the official implementation of the paper, "MARLP: Time-series Forecasting Control for Agricultural Managed Aquifer Recharge."

☀️ Horray! The trial of Year 2024 is finished! We will finalize the repo in April!

## 1 Abstract

The rapid decline of groundwater around the globe poses a significant challenge to sustainable agriculture. To address this issue, Agricultural Managed Aquifer Recharging (Ag-MAR) is proposed to recharge groundwater by artificially flooding agricultural lands using surface water. Ag-MAR requires a carefully selected schedule to avoid excessive flooding impacting the oxygen absorption of crop roots. However, current Ag-MAR scheduling fails to consider complex environmental factors such as weather and soil oxygen, resulting in either crop damage or insufficient recharging amount.

This paper proposes MARLP, the first end-to-end data-driven control system for Ag-MAR. We first formulate Ag-MAR as an optimization problem. To that end, we analyze four-year in-field datasets, which revealed the multi-periodicity feature of the soil oxygen level trends and the opportunity to use external weather forecasts as a clue for oxygen level prediction. Then, we design a two-stage forecasting framework. In the first stage, it extracts both the cross-variate dependency and the periodicity patterns from historical data, to conduct a preliminary forecasting. In the second stage, it leverages the weather-soil causal relationship and utilizes weather forecast data to facilitate accurate prediction of the soil oxygen levels. Finally, we use the prediction model to conduct model-predictive control (MPC) for Ag-MAR flooding. To tackle the challenge of large action space, we devise a heuristic planning module to reduce the number of flooding proposals to enable the search for optimal solutions. Real-world experiments show that MARLP improves the recharging amount in unit time by 42.67% compared with the previous four years while keeping an optimal oxygen deficit ratio.

## 2 Folder Organization

```plaintext
MARLP/

├── dataset/
│   ├── area_a.yml
│   └── area_b.yml

├── utils/ 
│   ├── **
│   └── utils.py
├── representation.py

├── control/ 
│   ├── **
│   └── utils.py
├── representation.py

├── wireless/ 
│   ├── **
│   └── utils.py
├── representation.py

```
