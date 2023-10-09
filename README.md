# Hackathon_Finance
# 🌟 코스피 단기예측 모델 🌟

## 🌐 개요
이 프로젝트는 랜덤 포레스트 기법을 사용하여 코스피 시장의 단기 방향성을 예측하는 모델을 개발하는 것을 목표로 하고 있습니다. 이 모델은 다양한 매크로 경제 지표와 시장 투자 지표들을 활용하여 코스피의 향후 3개월 방향성을 예측합니다.

### 📊 모델 예측 결과
모델 예측의 결과는 **‘상승’, ‘보합’, ‘하락’**의 3가지 투자 의견으로 나타납니다.

- **상승**: 향후 3개월 수익률이 4% 이상 기대될 때.
- **보합**: 향후 3개월 수익률이 0%~4% 사이로 기대될 때.
- **하락**: 향후 3개월 수익률이 0% 미만으로 기대될 때.

## 🚀 접근 방법
머신러닝 모델의 관점에서 볼 때, 과거 각 시점에서의 다양한 투자 지표들(예: WTI 유가, 원/달러 환율 등)의 데이터와 그 시점 사후의 3개월간 코스피 실제 수익률 데이터가 존재합니다. 이러한 데이터들은 지도 학습(supervised learning) 3항 분류 모델의 문제로 접근할 수 있으며, 각종 투자지표들을 피처(독립변수)로, 상승/보합/하락이라는 단기 방향성 결과를 레이블(종속변수)로 사용합니다.

## 🛠 모델 구성

### 알고리즘
랜덤 포레스트 모델

### 자료 기간
2006년 12월 30일부터 현재까지의 영업일별 시계열 자료. 약 4천 개.

### 피처(feature)
3종의 매크로, 밸류에이션, 어닝스 등의 투자지표 일별 수치. 예를 들어, WTI 유가, 원/달러 환율, VKOSPI.

### 레이블(label)
해당일의 사후 코스피 방향성. 상승/보합/하락으로 지정됨. 코스피의 사후 60영업일 수익률이 4% 이상이면 ‘상승’, 4%~0% 사이면 ‘보합’, 0% 미만이면 ‘하락’ 결과로 기록.

### 📈 향후 피처 추가 계획
향후에는 더 많은 피처들, 특히 일별 데이터를 추가할 계획입니다.

## 📅 가정
T일의 코스피 방향성은 ‘T+1일~T+61일 간의 사후 수익률’을 기준으로 삼습니다. 현실적인 투자 가능성을 위해 하루 뒤부터 투자하는 것으로 가정합니다.

## 📜 결론
이 README는 다양한 투자 지표를 고려하고 랜덤 포레스트 기법을 활용하여 코스피 시장의 단기 예측 모델에 대한 개요를 제공합니다. 이는 이 모델을 개발하는 데 있어서 작업 흐름, 피처, 레이블 및 가정을 이해하는 데 종합적인 가이드로 사용될 수 있습니다.

---
# 🌟 KOSPI Short-term Prediction Model 🌟

## 🌐 Overview
This project aims to develop a model to predict the short-term direction of the KOSPI market using the Random Forest technique. The model utilizes various macroeconomic indicators and market investment indicators to predict the direction of KOSPI over the next three months.

### 📊 Model Prediction Results
The results of model prediction are represented in three investment opinions: **'Up', 'Neutral', and 'Down'**.

- **Up**: Expected return is more than 4% in the next three months.
- **Neutral**: Expected return is between 0% and 4% in the next three months.
- **Down**: Expected return is less than 0% in the next three months.

## 🚀 Approach
From the machine learning model perspective, there exist actual data of various investment indicators (e.g., WTI oil prices, KRW/USD exchange rate) at each past time point, along with the actual KOSPI return data for the subsequent three months. The actual KOSPI return data can be converted into three levels: 'Up, Neutral, Down' based on the 4% / 0% criteria mentioned above.

This can be approached as a supervised learning ternary classification problem, using various investment indicators as features (independent variables) and short-term directional results as labels (dependent variables).

## 🛠 Model Configuration

### Algorithm
Random Forest Model

### Data Sample Period
Time series data from December 30, 2006, to the present, approximately 4000 daily samples.

### Features
Daily values of three types of investment indicators: macro, valuation, earnings, etc. For example, WTI oil prices, KRW/USD exchange rate, VKOSPI.

### Labels
The direction of KOSPI after the day, designated as 'Up/Neutral/Down'. The subsequent 60 business days' return of KOSPI is recorded as 'Up' if it's more than 4%, 'Neutral' if it's between 0% and 4%, and 'Down' if it's less than 0%.

### 📈 Future Feature Additions
More features, especially daily data, are planned to be added in the future.

## 📅 Assumptions
The direction of KOSPI on day T is based on the 'post-return between day T+1 and day T+61'. It is assumed to invest one day later for realistic investment possibilities.

## 📜 Conclusion
This README provides an outline of the short-term prediction model for the KOSPI market, considering various investment indicators and utilizing the Random Forest technique for prediction. It serves as a comprehensive guide to understanding the workflow, features, labels, and assumptions involved in developing this model.
---
