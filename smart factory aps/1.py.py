import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from warnings import filterwarnings
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import requests
from bs4 import BeautifulSoup
import pandas as pd
from lxml import etree
import json
import requests
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
import pprint
from pandas import DataFrame
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine
import re

plt.rc('font', family='NanumBarunGothic') # 한글폰트

import mysql.connector
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 데이터 입력
# MariaDB 서버 접속 정보 설정
host = 'localhost'
port = 3306
user = 'root'
password = 'maria'
database = 'my'

# MariaDB에 연결
conn = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)

try:
    cursor = conn.cursor()

    # 쿼리 실행 - 새로운 열 추가
    query = ""
    cursor.execute(query)

    # 변경사항을 커밋
    conn.commit()
    
    if conn.is_connected():
        
        select_query = 'select * from new_data where solddate <= "2021-04-25"'
        cursor.execute(select_query)
        selected_data = cursor.fetchall()
        
        if selected_data:
            for row in selected_data:
                SOLDDATE = row[0]
                PRODNAME = row[1]
                ORDER_QUANT = row[2]
                SOLD_QUANT = row[3]
                
                insert_query = "INSERT INTO dataset00 (SOLDDATE, PRODNAME, ORDER_QUANT, SOLD_QUANT) VALUES (%s, %s, %s, %s)"
                insert_data = (SOLDDATE, PRODNAME, ORDER_QUANT, SOLD_QUANT)
                
                cursor.execute(insert_query, insert_data)
                conn.commit()
                
    # 프로시저 호출
    cursor.callproc("Start0")
    conn.commit()
    print("create table successfully!\n")

except mysql.connector.Error as err:
    print("Error:", err)

finally:
    # 새로운 열이 추가된 테이블 조회
    query_select = "SELECT * FROM dataset00"
    df = pd.read_sql_query(query_select, conn)
    
    query_select2 = "SELECT * FROM pred_prot1"
    df0 = pd.read_sql_query(query_select2, conn)
    
    # 연결과 커서 닫기
    cursor.close()
    conn.close()

# 필요 열 선택 
df[['SOLDDATE', 'PRODNAME', 'SOLD_QUANT']]
RRP = df[['SOLDDATE', 'PRODNAME', 'SOLD_QUANT']]
co_df=RRP.copy()
co_df['SOLDDATE'] = pd.to_datetime(co_df['SOLDDATE'])

co_df.set_index('SOLDDATE',inplace=True)
scaler = MinMaxScaler()
co_df[['SOLD_QUANT']] = scaler.fit_transform(co_df[['SOLD_QUANT']])

# train, test 분할
std_date = co_df.index[-1]
testdate_start = std_date - relativedelta(months=6)
traindate_end = testdate_start - relativedelta(days=1)
traindate_start = co_df.index[0]

std_date1 = std_date
testdate_start1 = testdate_start
traindate_end1 = traindate_end
traindate_start1 = traindate_start

# # 모델학습 및 파라미터 설정 및 검증 결과 출력
# predicted_values_dic = {}
# pdq_li = []
# r2_li = []
# aic_li = []
# period = []
# r2_remove0_li = []

# for i in range(1):
#     print('train  :',traindate_start.date(),'~',traindate_end.date())
#     print('test   :',testdate_start.date(),'~',std_date.date())
#     r2_li1 = []
#     r2_remove0_li1 = []
#     aic_li1 = []
#     pdq_li1 = []
#     period.append(testdate_start.strftime('%y%m%d')+'~'+std_date.strftime('%y%m%d'))

#     for prod in co_df['PRODNAME'].unique():
#         series = co_df.query(f'PRODNAME=="{prod}"')
#         series.drop('PRODNAME',axis=1,inplace=True)
#         series = series.resample('W').sum()

#         # 데이터 분리

#         train = series['SOLD_QUANT'][traindate_start:traindate_end]
#         test = series['SOLD_QUANT'][testdate_start:std_date]


#         # Grid_search

#         p = d = q = range(0,2)
#         pdq = list(itertools.product(p,d,q))
#         seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
#         best_aic = np.inf
#         best_pdq = None
#         best_seasonal_pdq = None
#         tmp_model = None
#         best_mdl = None

#         for param in pdq:
#             for param_seasonal in seasonal_pdq:
#                 try:
#                     tmp_mdl = sm.tsa.statespace.SARIMAX(train, exog=None, order=param,
#                                                         seasonal_order = param_seasonal,
#                                                         enforce_stationarity=True,
#                                                         enforce_invertibility=True)
#                     res = tmp_mdl.fit()
#                     if res.aic < best_aic:
#                         best_aic = res.aic
#                         best_pdq = param
#                         best_seasonal_pdq = param_seasonal
#                         best_mdl = tmp_mdl
#                 except:
#                     continue

#         # 학습

#         res = sm.tsa.statespace.SARIMAX(endog=train, order=best_pdq,
#                                 seasonal_order = best_seasonal_pdq,
#                                 enforce_stationarity=True,
#                                 enforce_invertibility=True).fit()

#         aic_li1.append(res.aic)
#         pdq_li1.append((best_pdq, best_seasonal_pdq, best_aic))

#         try:
#             res.plot_diagnostics(figsize=(16,10))
#         except:
#             pass

#         # 시각화

#         pred = res.get_prediction(start=test.index.min(),
#                                 end = test.index.max(),
#                                 dynamic=True)

#         pred_ci = pred.conf_int()

#         # 검증(R2_score)

#         from sklearn.metrics import r2_score
#         predicted_value = pred.predicted_mean
#         r2 = r2_score(test,predicted_value)
#         r2_li1.append(r2)

#         # 모델, 결과값 저장

#         # res.save(f'D:\\hmkd1\\2차 프로젝트\\pkl\\{prod}{i}.pkl')
#         res.save(f'D:\\hmkd1\\2차 프로젝트\\pkl\\test_pkl2\\{prod}{i}.pkl')
#         # res.save(f'D:\\pythonscript\\2차 프로젝트\\pkl\\{prod}{i}.pkl')
#         try:
#             df = pd.concat([predicted_value,test],axis=1)
#             #   df.to_excel(f'D:\\hmkd1\\2차 프로젝트\\pkl\\{prod}{i}.xlsx')
#             df.to_excel(f'D:\\hmkd1\\2차 프로젝트\\pkl\\test_pkl2\\{prod}{i}.xlsx')
#             #   df.to_excel(f'D:\\pythonscript\\2차 프로젝트\\pkl\\{prod}{i}.xlsx')
#             predicted_values_dic[f'{prod}{i+1}'] = df
#             df1 = df[df['SOLD_QUANT']!=0]
#             r2_remove0_li1.append(r2_score(df1['SOLD_QUANT'],df1[0]))
#         except:
#             predicted_values_dic[f'{prod}{i}'] = predicted_value

#     std_date = testdate_start - relativedelta(weeks=1)
#     testdate_start = std_date - relativedelta(months=6)
#     traindate_end = testdate_start - relativedelta(days=1)
#     traindate_start = (traindate_end - relativedelta(years=2, months=3))
#     r2_li.append(r2_li1)
#     aic_li.append(aic_li1)
#     pdq_li.append(pdq_li1)
#     r2_remove0_li.append(r2_remove0_li1)

week_std_date = std_date1.to_pydatetime()
week_mon_date = week_std_date + timedelta(days = (7 - week_std_date.weekday()))
next_mon_date = week_mon_date.strftime('%Y-%m-%d')

month_date = week_mon_date + relativedelta(months=3)

# 모델 불러와서 제품별 예측량 뽑기
# 리스트에 있는 모든 제품들을 정의합니다.
products = co_df['PRODNAME'].unique()
result_dfs = []

for product in products:
    # 모델 불러오기
    with open(f'D:\\hmkd1\\2차 프로젝트\\pkl\\test_pkl2\\{product}0.pkl', 'rb') as file:
    # with open(f'D:\\pythonscript\\2차 프로젝트\\pkl\\test\\{product}0.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    forecast_steps = 56
    forecast = loaded_model.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    df = pd.DataFrame(forecast_values, columns=['predicted_mean'])
    df['week'] = df.index - pd.Timedelta(days=6)
    df = df.set_index('week')
    df.rename(columns={'predicted_mean': '예측중량'}, inplace=True)
    
    start_date = next_mon_date
    end_date = month_date 

    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    filtered_df['prodname'] = product

    # 정규화된 데이터에서 최소값과 최대값을 구합니다.
    min_val = RRP['SOLD_QUANT'].min()
    max_val = RRP['SOLD_QUANT'].max()

    # 정규화된 값을 원본 범위로 스케일링합니다.
    filtered_df['예측중량'] = (filtered_df['예측중량'] * (max_val - min_val)) + min_val
    
    # 결과 데이터프레임을 리스트에 추가
    result_dfs.append(filtered_df)

final_result_df = pd.concat(result_dfs)
final_result_df = final_result_df.reset_index()

# api
final_result_df['tem_avg'] = [None] * len(final_result_df)
final_result_df['hum_avg'] = [None] * len(final_result_df)
final_result_df['국내건설수주액'] = [None] * len(final_result_df)
final_result_df['국내기성액'] = [None] * len(final_result_df)

current_time = datetime.now()
# 현재시점을 불러옴
formatted__date = current_time.strftime("%Y%m")
# 건설api 형식으로 수정함 (년/월)
yesterday = current_time - timedelta(days=1)
# 어제시점을 불러옵니다 (기상청은 현재시점에 전날까지 업데이트)
w_date = yesterday.strftime("%Y%m%d")
# 기상청 api 형식으로 수정함(년/월/일)

data_dict = {
    '901Y104' : '건설기성액',
    '901Y020' : '국내건설수주액'
}

KEY = 'P6B82DH7S3NJPJF38DPV'
PERIOD = 'M'
START_DATE = '201811'
END_DATE = formatted__date

def get_product(KEY, STAT_CD, PERIOD, START_DATE, END_DATE):
    url = 'http://ecos.bok.or.kr/api/StatisticSearch/{}/xml/kr/1/1000/{}/{}/{}/{}/'.format(KEY, STAT_CD, PERIOD, START_DATE, END_DATE)

    response = requests.get(url).content.decode('utf-8')
    xml_obj = BeautifulSoup(response, 'lxml-xml')
    rows = xml_obj.findAll("row")
    return rows


item_list = [
    'STAT_CODE','STAT_NAME','ITEM_CODE1','ITEM_NAME1','ITEM_CODE2','ITEM_NAME2','ITEM_CODE3','ITEM_NAME3','UNIT_NAME',
    'TIME','DATA_VALUE'
]

result_list = list()

for k in data_dict.keys():
    rows = get_product(KEY , k ,PERIOD , START_DATE , END_DATE)

    for p in range(0, len(rows)):
        info_list = list()

        for i in item_list:
            try:
                individual_info = rows[p].find(i).text
            except:
                individual_info = ""
            info_list.append(individual_info)

        result_list.append(info_list)

result_df = pd.DataFrame(result_list , columns = ['통계표코드', '통계명', '통계 항목 1코드', '통계항목2코드', '통계항목명2', '통계항목3코드', '뭔데그럼', '나머지두개는', '단위', '시점', '값']).drop_duplicates()


gunsal_gisung_df = result_df[(result_df['통계명'] == '8.4.2. 건설기성액') &
                            (result_df['통계항목2코드'] == '총기성액') &
                            (result_df['통계항목3코드'] == '경상')].copy()

gunsal_suzu_df = result_df[(result_df['통계명'] == '8.4.1. 국내건설수주액') &
                          (result_df['통계항목2코드'] == '총수주액')].copy()

def extract_year_month(date_str):
    date_str = date_str.strftime('%Y-%m-%d')  # Timestamp를 문자열로 변환
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.year, date_obj.month

# autodf의 년월 추출
final_result_df['year'], final_result_df['month'] = zip(*final_result_df['week'].map(extract_year_month))

df0['week'] = pd.to_datetime(df0['week'])
df0['year'], df0['month'] = zip(*df0['week'].map(extract_year_month))

# 년월이 일치하는 경우 값을 할당
for _, row in final_result_df.iterrows():
    matching_row = gunsal_suzu_df[gunsal_suzu_df['시점'] == f"{row['year']}{row['month']:02d}"]
    if not matching_row.empty:
        final_result_df.loc[_, '국내건설수주액'] = matching_row['값'].values[0]

for _, row in final_result_df.iterrows():
    matching_row = gunsal_gisung_df[gunsal_gisung_df['시점'] == f"{row['year']}{row['month']:02d}"]
    if not matching_row.empty:
        final_result_df.loc[_, '국내기성액'] = matching_row['값'].values[0]

for _, row in df0.iterrows():
    matching_row = gunsal_suzu_df[gunsal_suzu_df['시점'] == f"{row['year']}{row['month']:02d}"]
    if not matching_row.empty:
        df0.loc[_, '국내건설수주액'] = matching_row['값'].values[0]

for _, row in df0.iterrows():
    matching_row = gunsal_gisung_df[gunsal_gisung_df['시점'] == f"{row['year']}{row['month']:02d}"]
    if not matching_row.empty:
        df0.loc[_, '국내기성액'] = matching_row['값'].values[0]





url2 = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
params = {
    'serviceKey': 'm6B9bWsl85oDiUgVP3Fkfrgdj89/ea59MvlX6XqTmaurYr2EhziN/Sii8RhhNULFgxQWj/DudnIBBN4rr1iWeg==',
    'pageNo': '1~80',
    'numOfRows': '900',
    'dataType': 'XML',
    'dataCd': 'ASOS',
    'dateCd': 'DAY',
    'startDt': '20210401',
    'endDt': w_date,   # 현재날짜에 하루전까지만 업데이트 됨
    'stnIds': '232' #천안232
}


w_response = requests.get(url2, params=params)



# 응답 데이터를 XML로 파싱
root = ET.fromstring(w_response.content)

# 데이터를 저장할 리스트 생성
data = []

# XML에서 데이터 추출하여 리스트에 추가
for item in root.iter('item'):
    TA_AVG = item.find('avgTa').text
    avgRhm = item.find('avgRhm').text
    tm = item.find('tm').text


    # 데이터를 딕셔너리 형태로 저장
    data.append({
        'tm': tm,
        'TA_AVG': TA_AVG,
        'avgRhm': avgRhm,

    })

# 데이터프레임 생성
df = pd.DataFrame(data)

for _, row in df.iterrows():
    matching_row = final_result_df[final_result_df['week']== row['tm']]
    if not matching_row.empty:
        final_result_df.loc[matching_row.index,'tem_avg'] = row['TA_AVG']
        final_result_df.loc[matching_row.index,'hum_avg'] = row['avgRhm']
        
for _, row in df.iterrows():
    matching_row = df0[df0['week']== row['tm']]
    if not matching_row.empty:
        df0.loc[matching_row.index,'tem_avg'] = row['TA_AVG']
        df0.loc[matching_row.index,'hum_avg'] = row['avgRhm']
        

# 1차 db작업

# MariaDB에 연결
conn = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)

try: 
    cursor = conn.cursor()
    
    for idx, row in df0.iterrows():
        week = row['week'].strftime('%Y-%m-%d')
        국내건설수주액 = row['국내건설수주액']
        국내기성액 = row['국내기성액']

        # UPDATE 쿼리 작성
        query_update = f"UPDATE pred_prot1 SET 국내건설수주액 = {국내건설수주액}, 국내기성액 = {국내기성액} WHERE week = '{week}';"

        # UPDATE 쿼리 실행
        cursor.execute(query_update)
        conn.commit()
    
    # 프로시저 호출
    cursor = conn.cursor()
    cursor.callproc("CreateAndCopyTables")
    conn.commit()
    print("Stored procedure executed successfully!\n")

    # final_result_df를 이용한 작업 시작
    for idx, row in final_result_df.iterrows():
        week = row['week'].strftime('%Y-%m-%d')
        order_quant = row['예측중량']
        prodname = row['prodname']
        tem_avg = row['tem_avg']
        hum_avg = row['hum_avg']
        국내건설수주액 = row['국내건설수주액']
        국내기성액 = row['국내기성액']
        
        query_insert = "INSERT INTO pred_prot1 (week, order_quant, prodname, tem_avg, hum_avg, 국내건설수주액, 국내기성액) \
               VALUES (%s, %s, %s, %s, %s, %s, %s);"
        values = (week, order_quant, prodname, round(float(tem_avg), 2), round(float(hum_avg), 2), 국내건설수주액, 국내기성액)
        cursor.execute(query_insert, values)
        conn.commit()
        
        # 쿼리 실행 - 새로운 열 추가
        query = ""
        cursor.execute(query)

        # 변경사항을 커밋
        conn.commit()

        # 새로운 열이 추가된 테이블 조회
        query_select = "SELECT week, prodname, tem_avg, hum_avg, order_quant, sold_quant, 국내건설수주액, 국내기성액, rn FROM pred_prot1"
        df2 = pd.read_sql_query(query_select, conn)
        
except mysql.connector.Error as err:
    print("Error:", err)

finally:
    # 연결과 커서 닫기
    cursor.close()
    conn.close()

# 생산예측
wpvna=df2['prodname'].unique()
wpvna=[wpvna]

std_date1 = std_date1.strftime('%Y-%m-%d')
testdate_start1 = testdate_start1.strftime('%Y-%m-%d')
traindate_end1 = traindate_end1.strftime('%Y-%m-%d')
traindate_start1 = traindate_start1.strftime('%Y-%m-%d')

# 사전(Dictionary)을 생성하여 각 제품명에 해당하는 데이터프레임을 저장할 것입니다.
product_df = {}
for product_name in wpvna[0]:
    # 특정 제품에 대한 데이터를 필터링합니다.
    filtered_data = df2[df2['prodname'] == product_name]
    filtered_data.drop(['rn','prodname'], axis=1, inplace=True)
    # 인덱스를 재설정하고 'week'를 인덱스로 설정합니다.
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data.set_index('week', inplace=True)
    filtered_data.rename(columns={'prodname':'제품명','order_quant':'예측중량','sold_quant':'판매수량'},inplace=True)
    scaler = MinMaxScaler()
    filtered_data[['tem_avg','hum_avg','국내건설수주액','국내기성액']] = scaler.fit_transform(filtered_data[['tem_avg','hum_avg','국내건설수주액','국내기성액']])
    filtered_data[['판매수량']] = np.log1p(filtered_data[['판매수량']])
    # 이제 filtered_data는 특정 제품에 대한 스케일링된 데이터를 포함합니다.
    # 이제 해당 제품명을 key로 하고 데이터프레임을 value로 하는 사전에 저장합니다.
    product_df[product_name] = filtered_data
    product_df[product_name].index = pd.to_datetime(product_df[product_name].index)

# product_models = {}

# for product_name in wpvna[0]:
#     filtered_data = product_df.get(product_name)

#     if filtered_data is not None:
#         y = filtered_data['판매수량']
#         X = filtered_data.drop('판매수량', axis=1)
#         X_train = X[traindate_start1:traindate_end1]
#         y_train = y[traindate_start1:traindate_end1]
#         X_test = X[testdate_start1:std_date1]
#         y_test = y[testdate_start1:std_date1]

#         ridge = Ridge(random_state=1004)
#         lasso = Lasso(random_state=1004)
#         dt_reg = DecisionTreeRegressor(random_state=1004)
#         rf_reg = RandomForestRegressor(random_state=1004)
#         xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=1004)
#         lgbm_reg = LGBMRegressor(random_state=1004)

#         ridge_parameters = {'alpha': [0.1, 0.5, 1.0]}
#         lasso_parameters = {'alpha': [0.0001, 0.0003, 0.0005]}
#         dt_parameters = {'max_depth': [3, 5, 7]}
#         rf_parameters = {'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000],
#                          'min_samples_split': [25, 50, 75, 100]}
#         xgb_parameters = {'n_estimators': [100, 200, 300, 400],
#                           'learning_rate': [0.01, 0.05, 0.1],
#                           'max_depth': [2, 4, 6, 8]}
#         lgbm_parameters = {'n_estimators': [100, 300, 500, 700, 800, 1000],
#                           'learning_rate': [0.01, 0.05, 0.1, 0.5],
#                           'colsample_bytree': [0.5, 0.75, 1.0]}

#         reg_param = [(ridge, ridge_parameters),
#                     (lasso, lasso_parameters),
#                     (dt_reg, dt_parameters),
#                     (rf_reg, rf_parameters),
#                     (lgbm_reg, lgbm_parameters),
#                     (xgb_reg, xgb_parameters)]

#         best_accuracy = -float('inf')
#         best_model = None
#         best_model_name = ""

#         for reg, parameter in reg_param:
#             grid_reg = GridSearchCV(reg, param_grid=parameter, scoring='neg_mean_squared_error', cv=3)
#             grid_reg.fit(X_train, y_train)

#             class_name = reg.__class__.__name__
#             scores_df = pd.DataFrame(grid_reg.cv_results_)

#             best_reg = grid_reg.best_estimator_
#             pred = best_reg.predict(X_test)

#             y_pred_expm = np.expm1(pred)
#             y_test_expm = np.expm1(y_test)

#             msle = mean_squared_error(y_test, pred)
#             rmsle = np.sqrt(msle)
#             accuracy = r2_score(y_test_expm, y_pred_expm)

#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_model = best_reg
#                 best_model_name = class_name

#         # 각 제품명별로 가장 적합한 모델을 product_models 사전에 저장합니다.
#         product_models[product_name] = best_model

# # 각 제품명별로 가장 적합한 모델을 파일로 저장합니다.
# for product_name, model in product_models.items():
#     # file_path = f"D:\\hmkd1\\2차 프로젝트\\pkl\\생산\\{product_name}_model.dat"
#     file_path = f"D:\\hmkd1\\2차 프로젝트\\pkl\\생산\\test2\\{product_name}_model.dat"
#     # file_path = f"D:\\pythonscript\\2차 프로젝트\\pkl\\생산\\{product_name}_model.dat"
#     joblib.dump(model, file_path)

df2_1 = df2[df2['week'] > std_date1]

product_df2 = {}
X_test_df = {}
predictions_by_product = {}
prediction_dfs = []

for product_name in wpvna[0]:
    filtered_data = df2_1[df2_1['prodname'] == product_name]
    filtered_data.rename(columns={'order_quant': '예측중량'}, inplace=True)
    scaler = MinMaxScaler()
    cols_to_scale = ['tem_avg', 'hum_avg', '국내건설수주액', '국내기성액']
    filtered_data[cols_to_scale] = scaler.fit_transform(filtered_data[cols_to_scale])
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data.set_index('week', inplace=True)
    product_df2[product_name] = filtered_data

    X_test_df[product_name] = product_df2[product_name].drop(['prodname', 'sold_quant', 'rn'], axis=1)
    grid_rf = joblib.load(f'D:\\hmkd1\\2차 프로젝트\\pkl\\생산\\test2\\{product_name}_model.dat')
    # grid_rf = joblib.load(f'D:\\pythonscript\\2차 프로젝트\\pkl\\생산\\test\\{product_name}_model.dat')
    pred_rf = grid_rf.predict(X_test_df[product_name])
    predictions_df = pd.DataFrame({'생산량': pred_rf}, index=X_test_df[product_name].index)
    predictions_df['생산량'] = np.ceil(np.expm1(predictions_df['생산량']))
    predictions_df.insert(0, 'prodname', product_name)
    predictions_df['week'] = predictions_df.index  # 인덱스를 week 컬럼으로 추가
    prediction_dfs.append(predictions_df)

all_predictions_df = pd.concat(prediction_dfs, ignore_index=True)
all_predictions_df = all_predictions_df.set_index('week').sort_index().reset_index()

# 2차 db작업

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

# raw_order 데이터프레임을 테이블로 저장
table_name = 'pred_prot3'  # 테이블 이름 지정
all_predictions_df.to_sql(table_name, con=engine, if_exists='replace', index=False)

print(f"Table '{table_name}' created successfully!")


# MariaDB에 연결
conn = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)
# 프로시저 호출
cursor = conn.cursor()
    
try:
    cursor.callproc("CreateAndCopyTables2")
    conn.commit()
    print("Stored procedure executed successfully!\n")

    # 쿼리 실행 - 새로운 열 추가
    query = ""
    cursor.execute(query)

    # 변경사항을 커밋
    conn.commit()
    
    # 새로운 열이 추가된 테이블 조회
    query_select = "SELECT * FROM pred_prot3"
    df3 = pd.read_sql_query(query_select, conn)

    # 쿼리 실행 - 새로운 열 추가
    query = ""
    cursor.execute(query)

    # 변경사항을 커밋
    conn.commit()

    # 새로운 열이 추가된 테이블 조회
    query_select2 = "SELECT * FROM new_re4"
    df4 = pd.read_sql_query(query_select2, conn)

except mysql.connector.Error as err:
    print("Error:", err)

finally:
    # 연결과 커서 닫기
    cursor.close()
    conn.close()
    
# 인벤 및 발주
df3.loc[df3['생산량'] <= 0, '생산량'] = 0
columns_to_drop = ['총중량','투입지시중량','제품여부']

df4.drop(columns_to_drop, axis=1, inplace = True)

result_dict = {}

# 각 날짜별로 레시피에 따라 원자재의 양을 계산하여 저장
for idx, row in df3.iterrows():
    week = row['week']
    prodname = row['prodname']
    생산량 = row['생산량']

    if week not in result_dict:
        result_dict[week] = {}

    # 해당 제품의 레시피 비율을 가져옴
    recipe =df4[df4['제품명'] == prodname]

    # 레시피에 따라 원자재의 양 계산
    for r_idx, r_row in recipe.iterrows():
        원자재명 = r_row['원자재명']
        비율 = r_row['비율']

        원자재_생산량 = round(생산량 * 비율, 2)

        if 원자재명 not in result_dict[week]:
            result_dict[week][원자재명] = 0

        result_dict[week][원자재명] += 원자재_생산량

result_data = []
for week, material_dict in result_dict.items():
    row = {'week': week}
    row.update(material_dict)
    result_data.append(row)

result_df = pd.DataFrame(result_data)
result_df.set_index('week', inplace=True)

row_count = len(result_df)
col_count = result_df.shape[1]
x=result_df.columns
x=[x]
re={}
for i in x[0]:
    re[i]=result_df[i][0]+result_df[i][1]

inven = pd.DataFrame([re], columns=re.keys())

larger_values = []
mean_values = []
safs = []

for i in range(row_count - 1):
    for col_index in range(col_count):  # 열 인덱스 범위로 순회합니다
        col_name = result_df.columns[col_index]  # 실제 열 이름을 가져옵니다
        larger_value = max(result_df[col_name][i], result_df[col_name][i+1])
        mean_value = (result_df[col_name][i] + result_df[col_name][i+1]) / 2
        saf = round((larger_value * 3) - (mean_value * 3), 2)

        larger_values.append(larger_value)
        mean_values.append(mean_value)
        safs.append(saf)

# 안전재고량 새로운 데이터프레임 생성
num_columns = col_count
data = {}  # 열 데이터를 저장할 딕셔너리

for i in range(num_columns):
    column_name = f'column_{i+1}'
    data[column_name] = []  # 빈 리스트로 초기화


# safs 리스트의 값을 데이터프레임 열에 할당
for i, value in enumerate(safs):
    column_name = f'column_{(i % num_columns) + 1}'  # 열 인덱스를 순환시킴
    data[column_name].append(value)

safe_inven = pd.DataFrame(data)
safe_inven.columns = result_df.columns
extracted_indices = result_df.index[:13]
safe_inven.index = extracted_indices

amt = []

# result_df.columns를 기준으로 루프 실행
for i in result_df.columns:
    # inven의 첫 번째 행 값
    inven_value = inven[i][0]

    # safe_inven의 각 행 값 더하기
    for j in range(len(safe_inven)):
        safe_inven_value = safe_inven[i][j]
        total_value = round(inven_value + safe_inven_value, 2)
        amt.append(total_value)

num_elements_per_row = row_count-1
result_matrix = []

while amt:
    row = amt[:num_elements_per_row]
    amt = amt[num_elements_per_row:]
    result_matrix.append(row)

for row in result_matrix:
    row

result_matrix = [[round(value, 2) for value in row] for row in result_matrix]
result_matrix2 = pd.DataFrame(result_matrix)
amt_df = result_matrix2.T
amt_df.columns = result_df.columns
extracted_indices = result_df.index[1:14]
amt_df.index = extracted_indices

result_df_aligned = result_df.reindex(index=safe_inven.index, columns=safe_inven.columns, fill_value=0)
raw_order = safe_inven + result_df_aligned

raw_order = raw_order.reset_index()

# 3차 db 작업

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

# raw_order 데이터프레임을 테이블로 저장
table_name = 'pred_prot5'  # 테이블 이름 지정
raw_order.to_sql(table_name, con=engine, if_exists='replace', index=False)

print(f"Table '{table_name}' created successfully!")

# MariaDB에 연결
conn = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)

# 커서 생성
cursor = conn.cursor()

try:
    # 프로시저 호출
    cursor.callproc("CreateAndCopyTables3")
    conn.commit()
    print("Stored procedure executed successfully!\n")

except mysql.connector.Error as err:
    print("Error:", err)

finally:
    # 연결과 커서 닫기
    cursor.close()
    conn.close()
