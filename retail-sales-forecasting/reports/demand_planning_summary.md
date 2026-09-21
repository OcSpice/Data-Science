# Demand Planning Run — Synthetic Retail Dataset

Generated from the 2023–2025 training data using the advanced demand-planning layer.

## Run snapshot

- 232,960 historical item-store-day observations
- 224 item-store demand series
- Planning baseline: recent 28-day mean
- Forecast-error proxy: 7-day seasonal-naive residuals
- Lead time assumption: 7 days
- Service level assumption: 95%
- Scenario multipliers: 0.90 / 1.00 / 1.10
- Simulated inventory position: 7 days of baseline demand

## Demand profile

| Demand segment | Series |
|---|---:|
| Low volume / Stable | 41 |
| Medium volume / Stable | 39 |
| High volume / Growing | 38 |
| High volume / Stable | 36 |
| Medium volume / Growing | 36 |
| Low volume / Growing | 34 |

## Scenario exposure

| Scenario | Total inventory | Total reorder requirement | Stockout-risk flags | Total inventory gap |
|---|---:|---:|---:|---:|
| Low demand | 9,753.75 | 11,097.60 | 224 | -1,343.85 |
| Base demand | 9,753.75 | 12,072.97 | 224 | -2,319.22 |
| High demand | 9,753.75 | 13,048.35 | 224 | -3,294.60 |

### Interpretation

Under the explicit simulation assumptions, the assumed seven-day inventory position is below the calculated reorder point for every item-store series in all three demand scenarios. This is a property of the simulated policy setup—not evidence of historical stockouts.

The high-demand scenario increases the aggregate reorder requirement relative to the base case, while the low-demand scenario reduces it. The result demonstrates how a forecast can be translated into inventory-policy scenarios even when observed inventory data is unavailable.

## Reproducibility

Run:

```bash
python -m advanced_demand_planning.src.run_planning
```

The detailed CSV outputs are generated under `retail-sales-forecasting/reports/`.
