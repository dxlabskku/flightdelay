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
data_flight_delayed = data_flight_parts[data_flight['label'] == 1]

# 3. Data Sampling
from sklearn.utils import resample
data_flight_normal = resample(data_flight_normal, n_samples='Number of delayed incidents', random_state=0)
data_flight = pd.concat([data_flight_normal, data_flight_delayed])

# 4. LSTM Model
import torch

def SetData(X, labels):  
    dataset = []
    dataset_y =[]
    tmp_A = list()
    test_number = 0
    idx = 0
    for t in X:
        tmp_B = list()
        error_flag = 0
        for i in range(1,60):        
            # 'target_time' is the time of the past when i is 1 hour ago and i is 2 hours ago when i is 2.
            # 'target_time_before_1hr' is the time 1 hour earlier than target_time.
            # 'target_time_after_1hr' is the time 1 hour later than target_time.
            # 'target_weather' is the weather data at target_time.
            # 'target_flight_data' is the flight history data at target_time.
            # If there is no flight history data at that time, we checked 1 hour before and after, 
            #   and if there is still no data, we considered it a normal flight.            
            target_time = pd.to_datetime(t) - i*(pd.to_datetime('01:00:00') - pd.to_datetime('00:00:00'))
            target_time_before_1hr = pd.to_datetime(t) - (i+1)*(pd.to_datetime('01:00:00') - pd.to_datetime('00:00:00'))
            target_time_after_1hr = pd.to_datetime(t) - (i-1)*(pd.to_datetime('01:00:00') - pd.to_datetime('00:00:00'))
            target_time = target_time.strftime('%Y-%m-%d %H')
            target_time_before_1hr = target_time_before_1hr.strftime('%Y-%m-%d %H')
            target_time_after_1hr = target_time_after_1hr.strftime('%Y-%m-%d %H')
            target_weather = data_weather[data_weather['YearMonthDayMin'] == target_time]
            x = target_weather[['temperature', 'dew_point', 'humidity', 'wind_speed', 'wind_gust', 
                                'pressure', 'precip', 'wind_CALM', 'wind_E', 'wind_ENE', 
                                'wind_ESE', 'wind_N', 'wind_NE', 'wind_NNE', 'wind_NNW', 
                                'wind_NW', 'wind_S', 'wind_SE', 'wind_SSE', 'wind_SSW', 
                                'wind_SW', 'wind_VAR','wind_W', 'wind_WNW', 'wind_WSW']]
            target_flight_data = data_flight_parts[data_flight_parts['YearMonthDayMin'] == target_time]
            if target_flight_data.empty:
                target_flight_data = data_flight_parts[data_flight_parts['YearMonthDayMin'] == target_time_after_1hr]
                if target_flight_data.empty:
                    target_flight_data = data_flight_parts[data_flight_parts['YearMonthDayMin'] == target_time_before_1hr]
                    if target_flight_data.empty:
                        x['Index100'] = 0   # normal flight : '0'
                    else:
                        x['Index100'] = int(target_flight_data['label'].iloc[-1])
                else:
                    x['Index100'] = int(target_flight_data['label'].iloc[-1])
            else:
                x['Index100'] = int(target_flight_data['label'].iloc[-1])
            try: 
                xx = x.iloc[0].to_list()
            except:
                error_flag = 1
                break
            tmp_B.append(xx)
        if error_flag == 0:
            tmp_B = torch.tensor(tmp_B)
            label = torch.tensor(labels[idx], dtype=torch.int)
            dataset.append(tmp_B)
            dataset_y.append(label)
            test_number += 1
        idx += 1
    return dataset, dataset_y

X = data_flight['YearMonthDayMin'].values
y = data_flight['label'].values
X_torch, y_torch = SetData(X, y)
X_torch = torch.stack(X_torch)
y_torch = torch.stack(torch_y)

import torch.utils.data as data
class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor,start=0, end=-1):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        self.start = start
        self.end = end
        if self.end == -1:
            self.end = len(x_tensor)
        
    def __getitem__(self, index):
        return self.x[index][self.start:self.end+1], self.y[index]

    def __len__(self):
        return len(self.x)
    
    from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LSTM(nn.Module):
    def __init__(self, device, input_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build_model()
        self.to(device)
        
    def build_model(self):
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = 1, dropout = 0.4, batch_first = True, bidirectional = False)
        self.fc = nn.Linear(self.hidden_dim, 'the number of layers') 
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss = nn.BCELoss() 
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)
        output = self.sigmoid(output)
        # print(output)
        return output.squeeze()
    
    def train(self, train_loader, valid_loader, epochs, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        
        loss_log = []
        for e in range(epochs):
            epoch_loss = 0
            for _, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(device=self.device, dtype=torch.float32)
                out = self.forward(data)
                loss = self.BCE_loss(out, target)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            loss_log.append(epoch_loss)
            
            valid_acc, valid_loss = self.predict(valid_loader)
            print(f'>> [Epoch {e+1}/{epochs}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_acc:.2f}% / Valid loss: {valid_loss:.4f}')
        return loss_log
    
    def predict(self,valid_loader, return_preds = False):
        BCE_loss = nn.BCELoss(reduction = 'sum')
        preds = []
        total_loss = 0
        correct = 0
        len_data = 0
        with torch.no_grad():
            for _, (data, target) in  enumerate(valid_loader):
                data, target = data.to(self.device), target.to(device=self.device, dtype=torch.float32)
                out = self.forward(data)
                len_data += len(target)
                loss = BCE_loss(out, target)
                total_loss += loss
                pred = (out>0.5).detach().cpu().numpy().astype(np.float32)
                preds += list(pred)
                correct += sum(pred == target.detach().cpu().numpy())
            acc = correct / len_data
            loss = total_loss/len_data
        if return_preds:
            return acc, loss, preds
        else:
            return acc, loss

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_torch,y_torch,stratify=y_torch,test_size=0.2,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,stratify=y_train,test_size=0.16,random_state=42)

test_list = [(1,3), (3,5), (7,9), (15,17), (23,25), (47,49)]
learning_rate_list = [0.0001, 0.0003, 0.0005, 0.001, 0.005]
num_epoch_list = [300, 400, 500, 600, 700]
for t_start, t_end in test_list:
    for l_r in learning_rate_list:
        for n_p in num_epoch_list:
            print(t_start,t_end,l_r,n_p)
            trainset = BasicDataset(X_train, y_train, t_start, t_end)
            validset = BasicDataset(X_valid, y_valid, t_start, t_end)
            testset = BasicDataset(X_test, y_test, t_start, t_end)
            batch_size = 32
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0)
            valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, num_workers=0)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), num_workers=0, shuffle = False)

            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            learning_rate = l_r 
            num_epoch = n_p
            input_dim = 26          # the number of X's features
            hidden_dim = 'the number of hidden dim'

            model = LSTM(device = device, input_dim=input_dim, hidden_dim=hidden_dim)
            model.to(device)
            model.train(train_loader = train_loader, valid_loader = valid_loader, epochs = num_epoch, learning_rate = learning_rate)

            from sklearn.metrics import classification_report
            y_test_re = []
            for i in range(len(testset)):
                y_test_re.append(int(testset[i][1]))
            acc, loss, p = model.predict(test_loader, return_preds = True)
            report = classification_report(y_test_re,p,digits=4, output_dict = True)
            result = pd.DataFrame(report).transpose()
            result.to_csv(f'Result_JFK_{t_start}_{t_end}_{l_r}_{n_p}.csv')
            result