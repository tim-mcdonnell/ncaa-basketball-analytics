---
title: NCAA Basketball Analytics - Model Performance Reference
description: Reference guide for baseline performance metrics and benchmarks for NCAA basketball prediction models
---

# Model Performance Reference

This document serves as a reference for expected model performance across different model types and prediction tasks in the NCAA Basketball Analytics project.

## Overview

The project uses several metrics to evaluate model performance:

- **Prediction Accuracy**: Percentage of correct win/loss predictions
- **Point Spread Accuracy**: Percentage of correctly predicted favorites/underdogs
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual point spreads
- **Calibration Error**: How well-calibrated probability predictions are
- **Brier Score**: Measures accuracy of probabilistic predictions

## Baseline Models

### Simple Baseline (Home Court Advantage)

The simplest baseline model predicts the home team wins by the average home court advantage.

| Metric | Performance |
|--------|-------------|
| Prediction Accuracy | 59.7% |
| Point Spread Accuracy | 59.7% |
| MSE | 194.6 |
| Calibration Error | 0.124 |
| Brier Score | 0.243 |

### Statistical Baseline (Elo Ratings)

A statistical baseline using Elo ratings.

| Metric | Performance |
|--------|-------------|
| Prediction Accuracy | 67.3% |
| Point Spread Accuracy | 68.1% |
| MSE | 132.4 |
| Calibration Error | 0.075 |
| Brier Score | 0.187 |

## Production Models

### Basic Game Prediction Model

The fundamental neural network model for game prediction.

| Metric | Performance | Improvement over Baseline |
|--------|-------------|---------------------------|
| Prediction Accuracy | 72.1% | +4.8% |
| Point Spread Accuracy | 73.4% | +5.3% |
| MSE | 97.2 | -35.2 |
| Calibration Error | 0.053 | -0.022 |
| Brier Score | 0.162 | -0.025 |

### Advanced Game Prediction Model

Enhanced neural network with additional features and more complex architecture.

| Metric | Performance | Improvement over Baseline |
|--------|-------------|---------------------------|
| Prediction Accuracy | 74.3% | +7.0% |
| Point Spread Accuracy | 75.8% | +7.7% |
| MSE | 87.5 | -44.9 |
| Calibration Error | 0.042 | -0.033 |
| Brier Score | 0.151 | -0.036 |

### Model Ensemble

Ensemble of multiple models with weighted aggregation.

| Metric | Performance | Improvement over Baseline |
|--------|-------------|---------------------------|
| Prediction Accuracy | 76.2% | +8.9% |
| Point Spread Accuracy | 77.5% | +9.4% |
| MSE | 83.1 | -49.3 |
| Calibration Error | 0.037 | -0.038 |
| Brier Score | 0.143 | -0.044 |

## Performance by Game Type

### Conference vs. Non-Conference Games

| Model Type | Conference Games Accuracy | Non-Conference Games Accuracy |
|------------|---------------------------|-------------------------------|
| Baseline (Elo) | 68.4% | 65.2% |
| Basic | 73.2% | 69.6% |
| Advanced | 75.1% | 72.5% |
| Ensemble | 77.0% | 74.6% |

### Tournament Games

| Model Type | Tournament Accuracy | Final Four Accuracy |
|------------|---------------------|---------------------|
| Baseline (Elo) | 63.1% | 57.8% |
| Basic | 67.3% | 63.9% |
| Advanced | 69.5% | 66.7% |
| Ensemble | 72.1% | 68.9% |

### Early Season vs. Late Season Games

| Model Type | Early Season (Nov-Dec) | Late Season (Feb-Mar) |
|------------|------------------------|----------------------|
| Baseline (Elo) | 62.9% | 69.8% |
| Basic | 67.2% | 74.6% |
| Advanced | 69.5% | 76.9% |
| Ensemble | 71.8% | 78.7% |

## Performance by Season

| Season | Baseline Accuracy | Basic Accuracy | Advanced Accuracy | Ensemble Accuracy |
|--------|-------------------|----------------|-------------------|-------------------|
| 2018-2019 | 66.9% | 71.3% | 73.8% | 75.7% |
| 2019-2020 | 67.4% | 72.0% | 74.1% | 76.0% |
| 2020-2021 | 65.8% | 70.7% | 73.2% | 75.1% |
| 2021-2022 | 68.1% | 73.0% | 75.1% | 77.0% |
| 2022-2023 | 67.8% | 72.5% | 74.6% | 76.5% |

## Calibration Analysis

### Win Probability Calibration by Model Type

| Predicted Win Probability | Baseline Actual % | Basic Actual % | Advanced Actual % | Ensemble Actual % |
|---------------------------|-------------------|----------------|-------------------|-------------------|
| 50-55% | 52.3% | 53.7% | 54.2% | 54.8% |
| 55-60% | 56.8% | 58.5% | 59.3% | 59.7% |
| 60-65% | 61.4% | 63.2% | 64.0% | 64.6% |
| 65-70% | 65.9% | 67.8% | 69.1% | 69.4% |
| 70-75% | 70.5% | 73.1% | 73.9% | 74.2% |
| 75-80% | 73.8% | 77.2% | 78.3% | 78.5% |
| 80-85% | 78.2% | 82.4% | 83.6% | 84.0% |
| 85-90% | 83.6% | 87.1% | 88.5% | 89.2% |
| 90-95% | 87.4% | 91.8% | 93.2% | 93.8% |
| 95-100% | 92.1% | 96.3% | 97.1% | 97.5% |

## Feature Importance

### Top 10 Features by Importance

| Feature | Basic Model Importance | Advanced Model Importance | Ensemble Importance |
|---------|------------------------|---------------------------|---------------------|
| Adjusted Offensive Efficiency | 17.2% | 15.8% | 16.3% |
| Adjusted Defensive Efficiency | 16.7% | 15.1% | 15.6% |
| Home Court Advantage | 9.3% | 7.2% | 8.1% |
| Effective Field Goal % | 8.1% | 7.6% | 7.8% |
| Turnover Rate | 7.5% | 7.0% | 7.2% |
| Offensive Rebound % | 7.2% | 6.8% | 6.9% |
| Free Throw Rate | 6.4% | 5.9% | 6.1% |
| Strength of Schedule | 5.8% | 8.4% | 7.4% |
| Tempo | 4.2% | 4.7% | 4.5% |
| 3-Point Shooting % | 3.9% | 5.2% | 4.7% |

## Tournament Prediction Performance

### NCAA Tournament Bracket Accuracy

| Season | Baseline Sweet 16 | Ensemble Sweet 16 | Baseline Elite 8 | Ensemble Elite 8 | Baseline Final 4 | Ensemble Final 4 |
|--------|-------------------|-------------------|------------------|------------------|------------------|------------------|
| 2019 | 9/16 | 11/16 | 4/8 | 5/8 | 1/4 | 2/4 |
| 2021 | 7/16 | 10/16 | 3/8 | 5/8 | 1/4 | 2/4 |
| 2022 | 8/16 | 12/16 | 3/8 | 5/8 | 2/4 | 3/4 |
| 2023 | 8/16 | 11/16 | 4/8 | 6/8 | 2/4 | 3/4 |

## Minimum Performance Requirements

For models to be promoted to production, they must meet the following minimum requirements:

| Metric | Minimum Requirement |
|--------|---------------------|
| Prediction Accuracy | ≥ 71.0% |
| Point Spread Accuracy | ≥ 72.0% |
| MSE | ≤ 100.0 |
| Calibration Error | ≤ 0.060 |
| Brier Score | ≤ 0.170 |
| Tournament Performance | ≥ 65% accuracy in test tournaments |

## Performance Degradation Monitoring

Models in production are monitored for performance degradation. Alerts are triggered if:

- Prediction accuracy drops below 70% over a 30-day window
- MSE increases above 105 over a 30-day window
- Calibration error increases above 0.065 over a 30-day window

## Model Retraining Schedule

| Model Type | Retraining Frequency | Triggers for Ad-hoc Retraining |
|------------|----------------------|--------------------------------|
| Basic | Monthly | Accuracy drops below 70% for 7 consecutive days |
| Advanced | Bi-weekly | Accuracy drops below 72% for 5 consecutive days |
| Ensemble | Weekly | Accuracy drops below 74% for 3 consecutive days |

## Versioning and Performance History

### Version History: Basic Model

| Version | Release Date | Prediction Accuracy | Point Spread Accuracy | MSE | Notable Changes |
|---------|--------------|---------------------|----------------------|-----|-----------------|
| 1.0 | 2022-10-01 | 70.2% | 71.3% | 102.4 | Initial release |
| 1.1 | 2022-11-15 | 71.4% | 72.5% | 99.3 | Improved feature preprocessing |
| 1.2 | 2023-01-10 | 72.1% | 73.4% | 97.2 | Enhanced early stopping |
| 1.3 | 2023-03-01 | 72.8% | 74.1% | 95.6 | Added team roster change features |

### Version History: Advanced Model

| Version | Release Date | Prediction Accuracy | Point Spread Accuracy | MSE | Notable Changes |
|---------|--------------|---------------------|----------------------|-----|-----------------|
| 1.0 | 2022-10-15 | 72.1% | 73.5% | 93.2 | Initial release |
| 1.1 | 2022-12-01 | 73.0% | 74.4% | 90.1 | Added player efficiency features |
| 1.2 | 2023-01-20 | 74.3% | 75.8% | 87.5 | Improved architecture with attention layers |
| 1.3 | 2023-03-15 | 75.0% | 76.5% | 85.2 | Added momentum features |

### Version History: Ensemble Model

| Version | Release Date | Prediction Accuracy | Point Spread Accuracy | MSE | Notable Changes |
|---------|--------------|---------------------|----------------------|-----|-----------------|
| 1.0 | 2022-11-01 | 73.8% | 75.0% | 88.5 | Initial release (3 models) |
| 1.1 | 2022-12-15 | 74.9% | 76.2% | 86.3 | Optimized ensemble weights |
| 1.2 | 2023-02-01 | 76.2% | 77.5% | 83.1 | Added two more models to ensemble |
| 1.3 | 2023-04-01 | 76.8% | 78.2% | 81.5 | Dynamic weighting based on game context |

## Conclusion

This performance reference will be updated quarterly as models are improved and more data becomes available. Teams working on model improvements should aim to exceed the current production model metrics listed in this document.

The NCAA Basketball Analytics team is currently prioritizing improvements in:

1. Early season game prediction accuracy
2. Tournament game prediction accuracy
3. Calibration for games with predicted win probabilities between 45-55%
