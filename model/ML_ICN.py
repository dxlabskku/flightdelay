# 1. Raw Data Load
import pandas as pd
data_weather = pd.read_csv('data_weather.csv', encoding = 'utf-8')
data_flight_normal = pd.read_csv('data_flight_normal.csv', encoding = 'utf-8')
data_flight_delayed = pd.read_csv('data_flight_delayed.csv', encoding = 'utf-8')

# 2. Data Pre-processing
data_weather_time_vector = data_weather['일시']
data_weather_time_vector = pd.to_datetime(data_weather_time_vector)
data_weather['time'] = data_weather_time_vector
data_weather['YearMonthDayMin'] = data_weather['time'].dt.strftime('%Y-%m-%d %H')
data_weather.columns = ['지점', '일시', 'Index1', 'Index2', 'Index3', 'Index4', 'Index5',
       'Index6', 'Index7', 'Index8', 'Index9', 'Index10', 'Index11', 'Index12', 'Index13',
       'Index14', 'Index15', 'Index16', 'Index17', 'Index18', 'Index19', 'Index20', 'Index21',
       'Index22', 'time', 'YearMonthDayMin']       
# 2.1. Data Linear Interpolation
data_weather['Index1'] = data_weather['Index1'].interpolate(method='linear')        
data_weather['Index2'] = data_weather['Index2'].interpolate(method='linear')
data_weather['Index4'] = data_weather['Index4'].interpolate(method='linear')
data_weather['Index5'] = data_weather['Index5'].interpolate(method='linear')
data_weather['Index6'] = data_weather['Index6'].interpolate(method='linear')
data_weather['Index7'] = data_weather['Index7'].interpolate(method='linear')
data_weather['Index8'] = data_weather['Index8'].interpolate(method='linear')
data_weather['Index10'] = data_weather['Index10'].interpolate(method='linear')
data_weather['Index11'] = data_weather['Index11'].interpolate(method='linear')
data_weather['Index12'] = data_weather['Index12'].interpolate(method='linear')
data_weather['Index14'] = data_weather['Index14'].interpolate(method='linear')
data_weather['Index15'] = data_weather['Index15'].interpolate(method='linear')
data_weather['Index16'] = data_weather['Index16'].interpolate(method='linear')
data_weather['Index18'] = data_weather['Index18'].interpolate(method='linear')
data_weather['Index19'] = data_weather['Index19'].interpolate(method='linear')
data_weather['Index20'] = data_weather['Index20'].interpolate(method='linear')
data_weather['Index21'] = data_weather['Index21'].interpolate(method='linear')
data_weather['Index22'] = data_weather['Index22'].interpolate(method='linear')

data_weather = data_weather.dropna(subset=['Index1', 'Index2', 'Index4', 'Index5', 'Index6', 'Index7', 
                                           'Index8', 'Index10', 'Index11', 'Index12', 'Index14', 'Index15', 
                                           'Index16', 'Index18', 'Index19', 'Index20', 'Index21', 'Index22'])
data_flight_normal = data_flight_normal.loc[data_flight_normal['계획'].str.contains(':')]
data_flight_normal = data_flight_normal.astype({'날짜': 'str', '계획': 'str'})
data_flight_normal_time_vector = data_flight_normal['날짜'] + " " + data_flight_normal['계획']
data_flight_normal['time'] = pd.to_datetime(data_flight_normal_time_vector)
data_flight_normal['YearMonthDayMin'] = data_flight_normal['time'].dt.strftime('%Y-%m-%d %H')
data_flight_normal['label'] = 0

data_flight_delayed = data_flight_delayed.loc[data_flight_delayed['계획'].str.contains(':')]
data_flight_delayed = data_flight_delayed.astype({'날짜': 'str', '계획': 'str'})
data_flight_delayed_time_vector = data_flight_delayed['날짜'] + " " + data_flight_delayed['계획']
data_flight_delayed['time'] = pd.to_datetime(data_flight_delayed_time_vector)
data_flight_delayed['YearMonthDayMin'] = data_flight_delayed['time'].dt.strftime('%Y-%m-%d %H')

data_flight_delayed['label'] = 0
for i in range(len(data_flight_delayed)):
    if data_flight_delayed['현황'][i] == '지연' or data_flight_delayed['현황'][i] == '출발':
        time1 = pd.to_datetime(data_flight_delayed['계획'][i])
        time2 = pd.to_datetime(data_flight_delayed['출발'][i])
        time_check = pd.to_datetime('01:00:00') - pd.to_datetime('00:00:00')
        if time2 - time1 >= time_check:
            data_flight_delayed['label'][i] = 1
    else:
        data_flight_delayed['label'][i] = 1

data_flight_delayed = data_flight_delayed[['항공사', 'time', 'YearMonthDayMin','label']] 
data_flight_normal = data_flight_normal[['항공사', 'time', 'YearMonthDayMin','label']] 

# 3. Data Sampling
from sklearn.utils import resample
data_flight_normal = resample(data_flight_normal, n_samples='Number of delayed incidents', random_state=0)
data_flight_normal = data_flight_normal[['항공사', 'time', 'YearMonthDayMin','label']] 
data_flight_delayed = data_flight_delayed[['항공사', 'time', 'YearMonthDayMin','label']] 
data_flight = pd.concat([data_flight_normal, data_flight_delayed])

time_difference_list = [(pd.to_datetime('02:00:00') - pd.to_datetime('00:00:00')), 
                        (pd.to_datetime('04:00:00') - pd.to_datetime('00:00:00')),
                        (pd.to_datetime('08:00:00') - pd.to_datetime('00:00:00')),
                        (pd.to_datetime('16:00:00') - pd.to_datetime('00:00:00')),
                        (pd.to_datetime('2022-01-02 00:00:00') - pd.to_datetime('2022-01-01 00:00:00')),
                        (pd.to_datetime('2022-01-03 00:00:00') - pd.to_datetime('2022-01-01 00:00:00'))
                       ]

# 4. Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np

for i in time_difference_list:
    time_difference = i
    new_time = pd.to_datetime(data_weather['YearMonthDayMin']) + time_difference
    data_weather['YearMonthDayMin'] = new_time.dt.strftime('%Y-%m-%d %H')
    data_total = pd.merge(data_flight, data_weather, on='YearMonthDayMin', how='left')
    
    X = data_total[['temperature', 'dew_point', 'humidity', 'wind_speed', 'wind_gust', 'pressure',
                    'precip', 'wind_CALM', 'wind_E', 'wind_ENE', 'wind_ESE', 'wind_N', 
                    'wind_NE', 'wind_NNE', 'wind_NNW', 'wind_NW', 'wind_S', 'wind_SE', 'wind_SSE', 
                    'wind_SSW', 'wind_SW', 'wind_VAR','wind_W', 'wind_WNW', 'wind_WSW',
                    'condition_Storm', 'condition_Freezing', 'condition_Windy',
                    'condition_Vicinity', 'condition_Haze', 'condition_Snow',
                    'condition_Fair', 'condition_Heavy', 'condition_Light',
                    'condition_Blowing', 'condition_Cloudy', 'condition_Wintry',
                    'condition_Squalls', 'condition_Hail', 'condition_Shallow',
                    'condition_Partly', 'condition_Rain', 'condition_Mostly',
                    'condition_Thunder', 'condition_Precipitation', 'condition_Patches',
                    'condition_Drizzle', 'condition_Mist', 'condition_Small', 'condition_Sleet', 'condition_Fog']].fillna(0)

    y = data_total['label']       
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DT
    from sklearn.model_selection import GridSearchCV
    dt = DecisionTreeClassifier(criterion= 'gini')
    dt.fit(X_train,y_train)
    prediction = dt.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_DT.csv')     

    # RF
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_RF.csv')     

    # SVM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC    
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(X_train, y_train)
    prediction = svm.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_SVM.csv')     
    
    # KNN
    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier()      
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_KNN.csv')     

    # LR
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings("ignore")
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_LR.csv')     
    
    # XGB
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_INCHEON_XGB.csv')     
