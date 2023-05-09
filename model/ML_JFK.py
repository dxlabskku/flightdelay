# 1. Raw Data Load
import pandas as pd
data_weather = pd.read_csv('data_weather.csv', encoding = 'utf-8')
data_flight = pd.read_csv('data_flight.csv', encoding = 'utf-8')

# 2. Data Pre-processing
data_weather_time_vector = data_weather['time']
data_weather_time_vector = pd.to_datetime(data_weather_time_vector)
data_weather['time'] = data_weather_time_vector
data_weather['YearMonthDayMin'] = data_weather['time'].dt.strftime('%Y-%m-%d %H')
data_weather.drop_duplicates(['YearMonthDayMin'], keep='first', inplace=True)
data_weather = data_weather[data_weather['wind'] != 0]
data_weather = data_weather[data_weather['wind_gust'] != 'mph']
data_weather.wind_speed = data_weather.wind_speed.astype(int)
data_weather.wind_gust = data_weather.wind_gust.astype(int)
data_weather.pressure = data_weather.pressure.astype(float)
data_weather.precip = data_weather.precip.astype(float)
data_weather.index = range(len(data_weather))
data_weather = data_weather.iloc[:, 1:] 
data_weather = pd.get_dummies(data_weather, columns =['wind'])

import re
condition_list = []
for i in range(len(data_weather)):
    condition_list.extend(re.findall("[a-zA-Z]+", data_weather['condition'][i]))    
condition_set = set(condition_list)
condition_list = list(condition_set)
condition_Series = pd.Series(condition_list)
condition_Series = condition_Series[~condition_Series.isin(['Unknown', 'N', 'A', 'Mix', 'T', 'of', 'with', 'in', 'the', 'and'])]
condition_list = condition_Series.tolist()

import numpy as np
data_weather['condition_Storm'] = np.where(data_weather['condition'].isin(["['Storm']"]), 1, 0) 
data_weather['condition_Freezing'] = np.where(data_weather['condition'].isin(["['Freezing']"]), 1, 0) 
data_weather['condition_Windy'] = np.where(data_weather['condition'].isin(["['Windy']"]), 1, 0) 
data_weather['condition_Vicinity'] = np.where(data_weather['condition'].isin(["['Vicinity']"]), 1, 0) 
data_weather['condition_Haze'] = np.where(data_weather['condition'].isin(["['Haze']"]), 1, 0) 
data_weather['condition_Snow'] = np.where(data_weather['condition'].isin(["['Snow']"]), 1, 0) 
data_weather['condition_Fair'] = np.where(data_weather['condition'].isin(["['Fair']"]), 1, 0) 
data_weather['condition_Heavy'] = np.where(data_weather['condition'].isin(["['Heavy']"]), 1, 0) 
data_weather['condition_Light'] = np.where(data_weather['condition'].isin(["['Light']"]), 1, 0) 
data_weather['condition_Blowing'] = np.where(data_weather['condition'].isin(["['Blowing']"]), 1, 0) 
data_weather['condition_Cloudy'] = np.where(data_weather['condition'].isin(["['Cloudy']"]), 1, 0) 
data_weather['condition_Wintry'] = np.where(data_weather['condition'].isin(["['Wintry']"]), 1, 0) 
data_weather['condition_Squalls'] = np.where(data_weather['condition'].isin(["['Squalls']"]), 1, 0) 
data_weather['condition_Hail'] = np.where(data_weather['condition'].isin(["['Hail']"]), 1, 0) 
data_weather['condition_Shallow'] = np.where(data_weather['condition'].isin(["['Shallow']"]), 1, 0) 
data_weather['condition_Partly'] = np.where(data_weather['condition'].isin(["['Partly']"]), 1, 0) 
data_weather['condition_Rain'] = np.where(data_weather['condition'].isin(["['Rain']"]), 1, 0) 
data_weather['condition_Mostly'] = np.where(data_weather['condition'].isin(["['Mostly']"]), 1, 0) 
data_weather['condition_Thunder'] = np.where(data_weather['condition'].isin(["['Thunder']"]), 1, 0) 
data_weather['condition_Precipitation'] = np.where(data_weather['condition'].isin(["['Precipitation']"]), 1, 0) 
data_weather['condition_Patches'] = np.where(data_weather['condition'].isin(["['Patches']"]), 1, 0) 
data_weather['condition_Drizzle'] = np.where(data_weather['condition'].isin(["['Drizzle']"]), 1, 0) 
data_weather['condition_Mist'] = np.where(data_weather['condition'].isin(["['Mist']"]), 1, 0) 
data_weather['condition_Small'] = np.where(data_weather['condition'].isin(["['Small']"]), 1, 0) 
data_weather['condition_Sleet'] = np.where(data_weather['condition'].isin(["['Sleet']"]), 1, 0) 
data_weather['condition_Fog'] = np.where(data_weather['condition'].isin(["['Fog']"]), 1, 0) 
data_weather = data_weather.dropna(subset=['temperature', 'dew_point', 'humidity', 'wind_speed', 'wind_gust',
                                            'pressure', 'precip',
                                            'wind_CALM', 'wind_E', 'wind_ENE', 'wind_ESE', 'wind_N', 'wind_NE',
                                            'wind_NNE', 'wind_NNW', 'wind_NW', 'wind_S', 'wind_SE', 'wind_SSE',
                                            'wind_SSW', 'wind_SW', 'wind_VAR', 'wind_W', 'wind_WNW', 'wind_WSW',
                                            'condition_Storm', 'condition_Freezing', 'condition_Windy',
                                            'condition_Vicinity', 'condition_Haze', 'condition_Snow',
                                            'condition_Fair', 'condition_Heavy', 'condition_Light',
                                            'condition_Blowing', 'condition_Cloudy', 'condition_Wintry',
                                            'condition_Squalls', 'condition_Hail', 'condition_Shallow',
                                            'condition_Partly', 'condition_Rain', 'condition_Mostly',
                                            'condition_Thunder', 'condition_Precipitation', 'condition_Patches',
                                            'condition_Drizzle', 'condition_Mist', 'condition_Small', 'condition_Sleet', 'condition_Fog'])

data_flight = data_flight.iloc[:, 1:] 
data_flight = data_flight[~data_flight['DepTime'].isna()]
data_flight.index = range(len(data_flight))
data_flight['WeatherDelay'] = data_flight['WeatherDelay'].fillna(0)

xx = data_flight['CRSDepTime'][:1]
for i in range(len(data_flight)):
    temp = data_flight['CRSDepTime'][i]         
    if temp == 2400:
        temp = 0
    temp2 = str(temp // 100) + ":" + str(temp % 100)    
    data_flight['CRSDepTime'][i] = temp2

data_flight_time_vector = data_flight['FlightDate'] + " " + data_flight['CRSDepTime']
data_flight_time_vector = pd.to_datetime(data_flight_time_vector)
data_flight['time'] = data_flight_time_vector
data_flight['YearMonthDayMin'] = data_flight_time_vector.dt.strftime('%Y-%m-%d %H')
data_flight['label'] = 0
for i in range(len(data_flight)):
    if data_flight['DepDelay'][i] >= 60 and data_flight['WeatherDelay'][i] != 0:
        data_flight['label'][i] = 1
    elif data_flight['DepDelay'][i] < 0:
        data_flight['label'][i] = 0
    else:
        data_flight['label'][i] = 2
data_flight = data_flight[data_flight['label'] != 2]
data_flight_parts =  data_flight[['Origin', 'YearMonthDayMin','label', 'time']]
data_flight_parts = data_flight_parts.sort_values('time', ascending=True)

data_flight_normal = data_flight_parts[data_flight['label'] == 0]
data_flight_abnormal = data_flight_parts[data_flight['label'] == 1]

# 3. Data Sampling
from sklearn.utils import resample
data_flight_normal = resample(data_flight_normal, n_samples='Number of delayed incidents', random_state=0)
data_flight = pd.concat([data_flight_normal, data_flight_abnormal])

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
    result.to_csv(f'Result_JFK_DT.csv')     

    # RF
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_JFK_RF.csv')     

    # SVM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC    
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(X_train, y_train)
    prediction = svm.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_JFK_SVM.csv')     
    
    # KNN
    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier()      
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_JFK_KNN.csv')     

    # LR
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings("ignore")
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_JFK_LR.csv')     
    
    # XGB
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    result = classification_report(y_test, prediction, digits=4, output_dict = True)
    result = pd.DataFrame(result).transpose()
    result.to_csv(f'Result_JFK_XGB.csv')     
