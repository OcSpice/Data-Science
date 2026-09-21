# Advanced Demand Planning & Inventory Simulation

This module extends the Retail Sales Forecasting project using the same synthetic M5-style dataset.

## Objective
Move from forecasting demand to using forecasts for operational planning.

The analysis answers:
1. Where is demand concentrated?
2. Which item-store series are stable, volatile, intermittent, growing, or declining?
3. How large is forecast uncertainty?
4. How might uncertainty affect reorder-point and safety-stock decisions?
5. Which products/stores create the greatest planning exposure?
6. What happens under low/base/high demand scenarios?

## Important data limitation
The current dataset contains sales, prices, events, calendar variables, and macro/weather features, but it does not contain observed inventory, supplier lead times, purchase orders, or stockout records.

Therefore this module does not claim to reconstruct historical inventory performance. Inventory outputs are explicitly scenario simulations based on configurable assumptions.

## Pipeline
Historical sales -> demand profiling -> segmentation -> forecast error -> uncertainty -> inventory scenario -> business simulation -> dashboard.

## Default assumptions
- Lead time: 7 days
- Service level: 95%
- Z-value: 1.645
- Safety stock uses historical forecast-error volatility
- Reorder point = expected lead-time demand + safety stock
- Scenario demand multipliers are configurable

## Outputs
- reports/demand_segments.csv
- reports/uncertainty_summary.csv
- reports/inventory_scenarios.csv
- reports/business_exposure.csv

The inventory layer is a simulation because the current dataset has no actual inventory or lead-time observations. Those assumptions will be replaced when real retail data becomes available.
