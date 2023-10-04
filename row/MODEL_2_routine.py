# -*- coding: utf-8 -*- 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.metrics as mt 
from sklearn.tree import export_graphviz 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV 
import joblib 
from sklearn.metrics import confusion_matrix 
t = {'temp':'temp'} 
# 1. 모델 로드
rnd_clf = joblib.load("forecast_model.pkl") 
print("\n< AI model: load >") 
# 2. new daily raw data 가져오기
# X, y는 최근일까지 포함한 전 데이터. X_past, y_past는 결과가 확인된 61일전까지의 데이터
model_data = pd.read_csv("1.csv")
X = model_data.iloc[:, 1:] 
X_names = X.columns 
y = model_data["forward_stage"] 
X_past = X[y.notna()] 
y_past = y[y.notna()] 
# 3. new daily raw data 전체 학습
rnd_clf.fit(X_past, y_past) 
print("\n< AI model: machine learning done >") 
print("accuracy_score of whole data: ", rnd_clf.score(X_past, y_past)) 
# 4. 현재(마지막) 데이터 표시
print("\n<Current status>")
for t['col'], t['score'] in zip(X.columns, X.iloc[-1]): 
 print("{:20} : {:>8.3f}".format(t['col'], t['score'])) 
X_current = np.array(X.iloc[-1]).reshape(1,-1) 
# 5. 현재 전망
print("\n< AI model: forecasting >") 
y_current_pred = rnd_clf.predict(X_current) 
print("forecast: ", y_current_pred) 
# 현재전망의 확률표
prob_current = rnd_clf.predict_proba(X_current) 
y_names = rnd_clf.classes_ 
print("\n[class] : [prob]") 
for t['name'], t['prob'] in zip(y_names, prob_current[0]): #prob_current[0]에 1개의 현재전망이 들어가기 때문에
 print("{:7} : {:.2f}".format(t['name'], t['prob'])) 
# 6. 2023년 일별 전망치의 확률 변화 
# 전기간 전망치 확률 데이터생성
prob = rnd_clf.predict_proba(X) 
prob_df = pd.DataFrame(prob, index=y.index, columns=y_names) 
# 2023년 전망치 확률의 그래프 출력
prob_2023 = prob_df['2023'] 
plt.bar(prob_2023.index, prob_2023['up'], label='up', bottom=prob_2023['neutral']+prob_2023['down'], 
color='r') 
plt.bar(prob_2023.index, prob_2023['neutral'], label='neutral', bottom=prob_2023['down'], color='g') 
plt.bar(prob_2023.index, prob_2023['down'], label='down', color='b') 
plt.legend() 
plt.show() 