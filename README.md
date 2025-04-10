# Load Forecasting Using Support Vector Machines (SVM)

This repository contains MATLAB code for forecasting electricity and gas loads for the year 2026 using Support Vector Machine (SVM) regression. The project was developed as part of a final-year engineering project at the University of Manchester, aiming to integrate machine learning techniques into multi-energy system simulations.

## Overview

The code trains individual SVM models for each power bus and gas node using 15 years of historical data (2011–2025). It applies a grid search over hyperparameters (BoxConstraint, Epsilon, and Kernel type) and selects the best model based on 5-fold cross-validation. Forecasted values for 2026 are compared against 2025 to ensure a realistic growth trend.

### Features

- Forecasts loads for:
  - 15 electricity buses (in MW)
  - 15 gas nodes (in MMCFD)
- Uses 5-fold cross-validation to select optimal hyperparameters
- Enforces 2026 ≥ 2025 forecast rule for stability
- Automatically selects alternative models if the best model predicts a decrease
- Plots two comparative graphs:
  - Historical and predicted power loads (2011–2026)
  - Historical and predicted gas loads (2011–2026)

## How to Use

1. Open the script file in MATLAB.
2. Ensure the `Statistics and Machine Learning Toolbox` is available.
3. Run the script directly.
4. View console outputs and generated plots.

## Output

- Forecasted 2026 and 2027 values for each location
- CV MSE and final MSE for each model
- Two figures:
  - Power Buses Load: Historical + 2026 Prediction
  - Gas Nodes Load: Historical + 2026 Prediction

## Prerequisites

- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox

## Files

- `SVM_forecasting.m` — Main script for training, forecasting, and plotting

## Author

Mohamed Wael Swelam  
University of Manchester

## License

For academic and research use only.
