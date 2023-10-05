import pandas as pd

# 파일 불러오기
file_path = '../merged_data.csv'  # 파일 경로를 필요에 따라 변경해주세요.
combined_data_df = pd.read_csv(file_path)

# 'Date' 열을 날짜 형식으로 변환
combined_data_df['날짜'] = pd.to_datetime(combined_data_df['날짜'])

# 데이터프레임의 기본 정보 출력
print("=== 데이터프레임의 기본 정보 ===")
print(combined_data_df.info())

# 기초 통계량 출력
print("\n=== 기초 통계량 ===")
print(combined_data_df.describe(include='all'))

# 결측치 정보 출력
missing_data = pd.DataFrame(combined_data_df.isnull().sum(), columns=['결측치 개수'])
missing_data['결측치 비율(%)'] = (missing_data['결측치 개수'] / len(combined_data_df)) * 100
print("\n=== 결측치 정보 ===")
print(missing_data)

# 각 열의 결측치 정보를 한국어로 출력
for col, row in missing_data.iterrows():
    print("{} 열의 결측치 개수는 {}개로, 전체 데이터의 {:.2f}% 입니다.".format(col, row['결측치 개수'], row['결측치 비율(%)']))
