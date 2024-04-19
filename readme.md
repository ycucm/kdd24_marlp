## MARLP: Time-series Forecasting Control for Agricultural Managed Aquifer Recharge

🌊 Welcome to the MARLP repository, featuring the official implementation of the paper, "MARLP: Time-series Forecasting Control for Agricultural Managed Aquifer Recharge."

☀️ Hooray! The trial of Year 2024 is finished! We will finalize the repo with a detailed instruction in April! 

## 1 Abstract

The rapid decline of groundwater around the globe poses a significant challenge to sustainable agriculture. To address this issue, Agricultural Managed Aquifer Recharging (Ag-MAR) is proposed to recharge groundwater by artificially flooding agricultural lands using surface water. Ag-MAR requires a carefully selected schedule to avoid excessive flooding impacting the oxygen absorption of crop roots. However, current Ag-MAR scheduling fails to consider complex environmental factors such as weather and soil oxygen, resulting in either crop damage or insufficient recharging amount.

This paper proposes MARLP, the first end-to-end data-driven control system for Ag-MAR. We first formulate Ag-MAR as an optimization problem. To that end, we analyze four-year in-field datasets, which revealed the multi-periodicity feature of the soil oxygen level trends and the opportunity to use external weather forecasts as a clue for oxygen level prediction. Then, we design a two-stage forecasting framework. In the first stage, it extracts both the cross-variate dependency and the periodicity patterns from historical data, to conduct a preliminary forecasting. In the second stage, it leverages the weather-soil causal relationship and utilizes weather forecast data to facilitate accurate prediction of the soil oxygen levels. Finally, we use the prediction model to conduct model-predictive control (MPC) for Ag-MAR flooding. To tackle the challenge of large action space, we devise a heuristic planning module to reduce the number of flooding proposals to enable the search for optimal solutions.

## 2 Dataset

Ag-MAR dataset contains data from 2020 to 2024. The files ended with '_wf_raw.csv' means it has weather forecast data.  The files ended with '_wf720.csv' means the timestamps are moved up for 720, to align the weather forecast input with historical input.
| Year | Flooding Duration | Sequence |
|------|-------------------|----------|
| 2020 | 2/20-4/2          | 6086     |
| 2021 | 2/12-3/31         | 6902     |
| 2022 | 1/19-4/8          | 11455    |
| 2023 | 2/28-4/6          | 5389     |
| 2024 | 2/8-4/7           | 8642     |

## 3 MPC Workflow

For each 10 minutes, the sensor data and weather forecasts are processed in real-time. Then the oxygen level curves are inferred for all flooding proposals. Accordingly, the best proposal would be selected and executed. This process constitutes a standardized control workflow.

## 4 Causal Projection Module

The causal projection module can leverage external forecasting results to calibrate the endogenous forecasting, as in Causal_Projection.py.

## Acknoledgement

This repository is built based on Time-series Library: https://github.com/thuml/Time-Series-Library. We plan to make pull request contributions to that repo.
