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
data_flight_parts = pd.concat([data_flight_delayed, data_flight_normal])
data_flight_parts = data_flight_parts.sort_values('time', ascending=True)

# 3. Data Sampling
from sklearn.utils import resample
data_flight_normal = resample(data_flight_normal, n_samples='Number of delayed incidents', random_state=0)
data_flight_normal = data_flight_normal[['항공사', 'time', 'YearMonthDayMin','label']] 
data_flight_delayed = data_flight_delayed[['항공사', 'time', 'YearMonthDayMin','label']] 
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
            x = target_weather[['Index1', 'Index2', 'Index4', 'Index5', 'Index6', 'Index7', 'Index8', 'Index10', 'Index11', 
                                'Index12', 'Index14', 'Index15', 'Index16', 'Index18', 'Index19', 'Index20', 'Index21', 'Index22']]
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
            input_dim = 19          # the number of X's features
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
            result.to_csv(f'Result_INCHEON_{t_start}_{t_end}_{l_r}_{n_p}.csv')
            result