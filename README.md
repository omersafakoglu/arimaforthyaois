# ARIMA Model for THYAO.IS Stock Prediction

## Project Overview

This project aims to predict THYAO.IS stock prices using the ARIMA (AutoRegressive Integrated Moving Average) model. By analyzing time series data, the project identifies trends and seasonality to forecast future stock prices.

## Table of Contents

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Development](#model-development)
- [Results](#results)
- [Development Ideas](#development-ideas)

## Getting Started

To run the project, follow these steps:
1. Clone this repository:  
   ```bash
   git clone https://github.com/omersafakoglu/arimaforthyaois.git
   
2.Install required libraries:
    ```bash
    pip install -r requirements.txt
    
3.Open the Jupyter Notebook and execute the code.

## Requirements

Python 3.6 or higher
statsmodels
pandas
numpy
matplotlib
scikit-learn

## Dataset

This project utilizes historical price data for the THYAO.IS stock. Data can be downloaded from Yahoo Finance or another reliable source.

## Model Development

The ARIMA model is developed through the following steps:

Data preprocessing.
Determining model parameters (p, d, q).
Training the model.
Making predictions.


## Results

Model performance is evaluated using the following metrics:

Mean Squared Error (MSE)
Symmetric Mean Absolute Percentage Error (SMAPE)
Results indicate the accuracy of the predictions and the overall performance of the model.
