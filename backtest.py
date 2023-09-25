# -*- coding: utf-8 -*- 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# -1. MDD 함수 정의
def MDD(list_values): 
 mdd_value = 0 
 for i in range(1, len(list_values)): 
    bw_max = max(list_values[:i]) 
    curr = list_values[i] 
    mdd = curr / bw_max - 1 
 if mdd < mdd_value: 
    mdd_value = mdd 
 return mdd_value 
# 0. 사후방향성 클래스를 수치로 전환하는 함수 정의 (up=1, neutral=0, down=-1) 
def convert_num(pred): 
    pred_num = np.empty(len(pred)) 
    pred_num[pred=='up']=1 
    pred_num[pred=='neutral']=0 
    pred_num[pred=='down']=-1 
    pred_num[pred.isna()]=np.NaN 
    
    return pred_num 
# 1. 월간 시장수익률 데이터 가져오기
# 데이터는 일자/월수익률(원수치)/코스피지수/월말하루전영업일로 구성
kdata = pd.read_csv("investing.csv")
# 2. AI모델로 예측한 예측정보를 가져오기
# 기존 AI모델 학습 프로세스에서 만들어낸 y_pred 데이터를 조회함. 
# 매월말에 예측했던 투자의견을 가져옴. 단, 실제 투자를 위해서, 월말하루전영업일 기준 자료를 가져옴
# kdata['stage']='' 
kdata['stage'] = y_pred #말일자 아님
for index in kdata.index: 
   kdata.loc[index,'stage']=y_pred[kdata.loc[index,'before_last']] 
# 3. 전월말 투자의견 열 생성
kdata['pre_stage']= kdata['stage'].shift(1) 
kdata['port_return']=0 
# 4. 전략 수익률 생성
# 전월말 투자의견이 상승이면 코스피 long, 보합이면 Cash, 하락이면 코스피 Short 실행. 
# 해당 전략에 따라 포트 월별수익률(port_return) 생성
kdata.loc[kdata['pre_stage']=='up', 'port_return'] = kdata['m_return']*1 
kdata.loc[kdata['pre_stage'].isna(), 'port_return'] = kdata['m_return']*1 
kdata.loc[kdata['pre_stage']=='neutral', 'port_return'] = 0 
kdata.loc[kdata['pre_stage']=='down', 'port_return'] = kdata['m_return']*-1 
# 코스피와 모델포트폴리오의 누적수익률(1에서 시작하는 인덱스 형태) 생성
kdata['kospi_cumul']=(1+kdata['m_return']).cumprod() 
kdata['port_cumul']=(1+kdata['port_return']).cumprod() 
# 5. 백테스팅 결과 기록(CAGR, 변동성, Sharpe ratio, MDD) 
my_back = {'months':len(kdata)} 

my_back['k_cumul_return_idx']=kdata['kospi_cumul'][-1] 
my_back['k_cumul_return_pct']=(my_back['k_cumul_return_idx']-1)*100 
my_back['k_cagr']=(my_back['k_cumul_return_idx']**(12/my_back['months']))-1 
my_back['k_cagr_pct']=my_back['k_cagr']*100 
my_back['k_vol_pct']=np.std(kdata['m_return'])*np.sqrt(12)*100 
my_back['k_Sharpe']=my_back['k_cagr_pct']/my_back['k_vol_pct'] 
my_back['k_MDD']=MDD(kdata['kospi_cumul'])*100 

my_back['port_cumul_return_idx']=kdata['port_cumul'][-1] 
my_back['port_cumul_return_pct']=(my_back['port_cumul_return_idx']-1)*100 
my_back['port_cagr']=(my_back['port_cumul_return_idx']**(12/my_back['months']))-1 
my_back['port_cagr_pct']=my_back['port_cagr']*100 
my_back['port_vol_pct']=np.std(kdata['port_return'])*np.sqrt(12)*100 
my_back['port__Sharpe']=my_back['port_cagr_pct']/my_back['port_vol_pct'] 
my_back['port_MDD']=MDD(kdata['port_cumul'])*100 

# 6. 백테스팅 결과 출력하기
print("<Backtesting result>") 
for key, value in my_back.items(): 
 print("{:22}: {:>8.3f}".format(key, value)) 
 
# 포트폴리오 누적수익률 그래프
kdata['port_cumul'].plot() 
plt.title('Portfolio performance index') 
plt.ylabel('\'02/12/31 = 1') 
plt.show() 

# 7. 월말 모델전망치와 실제결과치 출력
# 월말 실제결과치 입력
for index in kdata.index: 

 kdata.loc[index,'real_stage']=y[kdata.loc[index,'before_last']] 
# 사후방향성 클래스를 수치로 변환
kdata['stage_num']=convert_num(kdata['stage']) 
kdata['real_stage_num']=convert_num(kdata['real_stage']) 

# 전망치와 결과치의 그래프 출력
kdata.plot(y=['stage_num', 'real_stage_num'], label=['model forecast','real direction']) 
plt.title('Model forecast vs. Real direction') 
plt.ylabel('up=1, neutral=0, down=-1') 
plt.show() 