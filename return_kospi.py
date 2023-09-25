import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

# 코스피 200 지수의 티커 심볼을 정의합니다.
symbol = "^KS11"

# yfinance로부터 데이터를 불러옵니다.
kospi_data = yf.download(symbol, start="2006-12-30")

# 월말을 기준으로 데이터를 재샘플링합니다.
kospi_resampled = kospi_data['Adj Close'].resample('M').last()

# 월간 수익률을 계산합니다.
kospi_resampled_returns = kospi_resampled.pct_change().dropna()

# 필요한 열들에 대해 새로운 DataFrame을 생성합니다.
kospi_df = pd.DataFrame(index=kospi_resampled.index)
kospi_df['kospi'] = kospi_resampled
kospi_df['m_return'] = kospi_resampled_returns
kospi_df['base_date'] = kospi_data.index + BMonthEnd(0)  # 각 날짜에 대해 당월말을 얻습니다.

# 'before_last' - 월말 하루 전 영업일을 계산합니다.
before_last = kospi_data.index.searchsorted(kospi_df.index) - 1  # 각 월말의 인덱스를 찾아 그 전날의 인덱스를 얻습니다.
kospi_df['before_last'] = kospi_data.index[before_last]

# 데이터 프레임 컬럼과 형식을 재구성합니다.
formatted_kospi_df = pd.DataFrame()
formatted_kospi_df['base_date'] = kospi_df['base_date'].dt.strftime('%Y-%m-%d')  # 날짜를 'YYYY-MM-DD' 형식으로 변환
formatted_kospi_df['m_return'] = kospi_df['m_return'].round(3)  # m_return을 소수점 세 자리까지 반올림
formatted_kospi_df['kospi'] = kospi_df['kospi'].round(2)  # kospi를 소수점 둘째 자리까지 반올림
formatted_kospi_df['before_last'] = kospi_df['before_last'].dt.strftime('%Y-%m-%d')  # 날짜를 'YYYY-MM-DD' 형식으로 변환

# 형식이 지정된 데이터 프레임을 인덱스 없이 CSV 파일로 저장합니다.
filename = "return_kospi_correct.csv"
formatted_kospi_df.to_csv(filename, index=False)
