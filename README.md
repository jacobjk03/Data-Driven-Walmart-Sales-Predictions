# Data-Driven Walmart Sales Predictions

## Project Overview

This project focuses on predicting store-level sales at Walmart using time series forecasting models. Accurate sales predictions are critical for managing inventory, optimizing business metrics like Gross Margin and Average Order Value (AOV), and enhancing customer satisfaction. By leveraging multiple machine learning and statistical models, this project demonstrates a significant improvement in forecast accuracy, leading to optimized business performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Analyzed](#models-analyzed)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project is from the [M5 Forecasting Accuracy competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy). It contains sales data from Walmart across various departments and stores, as well as calendar and price data that influence demand. The goal is to predict sales for 28 days in the future at the store-item level.

## Models Analyzed

We explored various models to identify the best fit for accurate sales forecasting:
- **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
- **Exponential Smoothing**
- **LSTM (Long Short-Term Memory)**
- **Random Forest**
- **LightGBM**

Each model was carefully evaluated based on its Root Mean Square Error (RMSE), with Exponential Smoothing emerging as the best performer, achieving a 38.92% improvement in RMSE compared to LSTM.

## Results

- **Exponential Smoothing** showed the best results with a **38.92% improvement in RMSE** over LSTM.
- Optimized inventory management led to improvements in key business metrics such as:
  - **Gross Margin**
  - **Average Order Value (AOV)**
  - **Customer Satisfaction**
- These improvements enhanced sales prediction accuracy, which in turn, led to better stock planning and increased business performance.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: 
  - Data Analysis: `pandas`, `NumPy`
  - Visualization: `matplotlib`
  - Machine Learning: `scikit-learn`, `TensorFlow`, `Keras`
  - Statistical Modeling: `statsmodels`
  
## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/jacobjk03/Data-Driven-Walmart-Sales-Predictions.git
    ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) and place it in the `data/` folder.


## Conclusion

By analyzing and fine-tuning different models, the project successfully improved Walmart’s sales forecasting, resulting in more effective inventory management and improved business outcomes. The best performance was achieved using Exponential Smoothing, demonstrating its capability in time series forecasting.

## Acknowledgments

- The dataset is provided by the [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition on Kaggle.
- Special thanks to Walmart for their anonymized sales data.
