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
# 1. daily raw data 가져오기
model_data = pd.read_csv("1.csv", sheet_name="raw", header=18, index_col=0) 
# 2. features, label 전체데이터 생성
# X, y는 최근일까지 포함한 전 데이터. X_past, y_past는 결과가 확인된 61일전까지의 데이터
X = model_data.iloc[:, 1:] 
X_names = X.columns 
y = model_data["forward_stage"] 
X_past = X[y.notna()] 
y_past = y[y.notna()] 
# 3. train, test 나누기
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X_past, y_past): #sss.split(~) 안에 n_splits 수만큼 준비됨
 X_train, X_test = X_past.iloc[train_index,], X_past.iloc[test_index,] 
 y_train, y_test = y_past[train_index], y_past[test_index] 
# ===== 랜덤포레스트 메인 ===== 
# 4. 모델 세부 튜닝: 최적 하이퍼파라미터 찾기
rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42) 

param_dist_rf = { 
'n_estimators':[50, 100, 500], 
'max_leaf_nodes':[20, 30, 40, 50], 
'max_features':[2, 4, 6, 8] 
} 

rnd_search = RandomizedSearchCV(rnd_clf, param_dist_rf, cv=10, random_state=42) 
rnd_search.fit(X_train, y_train) 
print(rnd_search.best_params_) 
# 5. 학습 및 K-fold cross_validation 평가 
rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=40, max_features=8, n_jobs=-1, 
random_state=42) #디폴트
rnd_scores = cross_val_score(rnd_clf, X_train, y_train, scoring="accuracy", cv=10) 
print("\n<10-fold cross-validation>") 
print("accuracy score mean: ", rnd_scores.mean()) 
# 6. 최종 모델 학습
rnd_clf.fit(X_train, y_train) 
print("\n<AI model: machine learning done >") 
print("accuracy_score of train data(0.8 of sample): ", rnd_clf.score(X_train, y_train)) 
# 7. test data 확인
print("accuracy_score of test data(0.2 of sample): ", rnd_clf.score(X_test, y_test)) 
#y_test_pred = rnd_clf.predict(X_test) 
#print("accuracy_score of test data: ", mt.accuracy_score(y_test, y_test_pred)) 
# 8. confusion matrix 확인
y_test_pred = rnd_clf.predict(X_test) 
cm1= confusion_matrix(y_test, y_test_pred, labels=["up","neutral","down"]) 
print("\n<Confusion matrix>") 
print("(of test)") 
print("up","neutral","down") 
print(cm1) 
cm2= confusion_matrix(y_past, rnd_clf.predict(X_past), labels=["up","neutral","down"]) 
print("(of all)") 
print("up","neutral","down") 
print(cm2) 
# 9. 변수 중요도 체크
print("\n<Feature importance>") 
for name, score in zip(X.columns, rnd_clf.feature_importances_): 
 print(name, ": ", score) 
# 10. backtesting용 과거의 예측데이터 생성
y_prediction = rnd_clf.predict(X) 
y_pred = pd.Series(y_prediction, index=y.index) 
# 11. 모델 저장
joblib.dump(rnd_clf, "forecast_model.pkl") 
print("\n< AI model: save >") 