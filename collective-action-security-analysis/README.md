# Patterns of Collective Action and Civilian Security in Africa (2024–2025)

## Project overview

This project is a comparative data analytics and research project using event-level conflict and demonstration data to examine patterns of collective action and civilian security across African countries.

It is deliberately different from the retail forecasting project:
- Retail forecasting: predictive analytics, time-series forecasting, model validation and demand planning.
- This project: descriptive and diagnostic analytics, geographic analysis, normalization, segmentation, comparative reporting and dashboard design.

The goal is not to predict political events or make political recommendations. The goal is to turn large event-level data into transparent, reproducible evidence that can be explored by country, event type, time period and impact.

## Analytical questions

1. How are event types distributed across countries and time?
2. Which countries and regions record the highest event volumes after accounting for population?
3. How do demonstrations, political violence and civilian-targeting events differ in frequency and geographic distribution?
4. How do event counts and fatalities change between 2024 and 2025?
5. Which countries show high event frequency but comparatively different levels of civilian impact?
6. How does population-normalized activity change the interpretation of raw event counts?
7. What geographic and temporal patterns are visible in the data?

## Core analytical capabilities

### Data Analytics
- Data cleaning and validation
- Exploratory data analysis
- Aggregation and grouping
- KPI design
- Ratio and rate calculations
- Comparative analysis
- Trend analysis
- Geographic analysis
- Outlier investigation
- Research storytelling

### Data Science
- Feature engineering
- Event classification
- Population normalization
- Segmentation
- Exploratory relationship analysis
- Reproducible Python analysis

### BI / Visualization
- Power BI dashboard
- Interactive country and year filters
- Geographic visualization
- KPI cards
- Trend charts
- Comparative analysis without political scoring

### Data engineering / modeling
- Fact/dimension analytical model
- Reusable Python transformations
- Data-quality checks
- Reproducible reporting

## Planned data model

Fact table: fact_events
- event date
- country
- region
- location
- event type
- sub-event type
- actor/category fields
- fatalities
- civilian-targeting indicator where available
- event count

Dimensions:
- dim_date: date, year, month, quarter
- dim_country: country, region, population year, population
- dim_event_type: event type, sub-event type, analytical category
- dim_location: country, admin-1/state/province, city/locality, latitude, longitude

## Key metrics

### Event volume
Total Events = count of recorded event rows.

### Fatalities
Total Fatalities = sum of recorded fatalities.

### Events per 100,000 people
(Total Events / Population) × 100,000

### Fatalities per 100,000 people
(Total Fatalities / Population) × 100,000

### Civilian-targeting share
Civilian-targeting Events / Total Events, where the source classification supports this measure.

All rate calculations will clearly identify their denominator and reference year.

## Planned workflow

Raw event data → data quality checks → cleaning and standardization → event classification → country/location enrichment → population normalization → EDA → Power BI model → dashboard → research-style findings.

## Planned deliverables

- data/ — source-data documentation and processed-data notes
- src/ — reusable Python cleaning and analysis code
- notebooks/ — exploratory analysis
- sql/ — analytical SQL queries
- dashboard/ — Power BI/dashboard documentation
- reports/ — findings, methodology and data dictionary

## Dashboard structure

### Page 1 — Executive overview
- Total events
- Total fatalities
- Events per 100k
- Fatalities per 100k
- Year comparison
- Country/region map

### Page 2 — Event patterns
- Event type distribution
- Monthly/quarterly trends
- Country comparison
- Sub-event breakdown

### Page 3 — Civilian impact
- Civilian-targeting share
- Fatalities by country
- Fatalities over time
- Event type vs. fatalities

### Page 4 — Geographic analysis
- Country-level map
- Subnational concentration
- Location-level event distribution
- Filters for country, year and event type

### Page 5 — Comparative analysis
- Raw counts vs. population-normalized rates
- 2024 vs. 2025 comparison
- Country profiles
- Key analytical observations

## SQL component

The project will include SQL queries for monthly event trends, country-level event totals, fatalities by event type, population-normalized rates, year-over-year changes, civilian-targeting analysis, top locations and country profiles.

## Data integrity principles

- Preserve original event records where possible.
- Document transformations.
- Separate raw, cleaned and analytical data.
- Avoid unsupported causal claims.
- Distinguish event frequency from severity.
- Distinguish raw counts from population-normalized measures.
- Document missing-data treatment.
- Avoid treating correlation as causation.

## Important interpretation note

Event counts, fatalities and civilian-impact indicators describe recorded events in the source data. They do not by themselves establish causation, intent, or the full level of activity in a country. Coverage, reporting practices, missingness and classification rules can affect observed patterns.

## Portfolio positioning

| Project | Primary capability |
|---|---|
| Retail Sales Forecasting & Demand Planning | Predictive analytics, time series, ML, inventory decisions |
| Collective Action & Civilian Security | Descriptive/diagnostic analytics, geospatial analysis, normalization, BI storytelling |

Together, the projects demonstrate two different sides of analytics: predicting future demand and explaining patterns in complex event data.