# Predicting flight departure delays caused by weather conditions using machine learning and deep learning analysis

This paper focused on predicting flight departure delays using machine learning and deep learning techniques. Flight departure delays have become a common issue. Several researchers have proposed various technologies and solutions for airline prediction, but the data collection period and the pre-flight delay prediction time were limited. Therefore, This paper performed a study on predicting flight delays based on 11 years of long-term weather data using machine learning and deep learning techniques for six time differences (2hr, 4hr, 8hr, 16hr, 24hr, 48hr). This paper focused on three airports: Incheon International Airport in South Korea, John F. Kennedy International Airport, and Chicago Midway International Airport in the United States. To achieve high prediction accuracy, several feature extraction methods were adopted, and various machine learning and deep learning models were employed. The experimental results showed that with a 2-hour time difference, the RF model reported the highest accuracy score of 0.749 for ICN airport, the LSTM model achieved the highest accuracy score of 0.852 for JFK airport, and the LSTM model achieved the highest accuracy score of 0.785 for MDW airport.

- We show that it was possible to predict flight takeoff delays based on weather data. 
  By utilizing a long period of data, a model that is suitable for general situations could be constructed.
- This model could be applied to various fields such as ocean vessel delays due to weather, 
  vehicle operation restrictions due to weather, and outdoor construction work stoppages due to weather.
- Through early warning in these application areas, 
  it is possible to prepare for potential human and property damage.

# Data
We collected weather datasets from the following sources: 
for Incheon International Airport from [the Korea Meteorological Administration's Open Data Portal] (https://data.kma.go.kr/data/air/selectAmosRltmList.do?pgmNo=575&tabNo=1), 
for New York City from [Weather Underground] (https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA), 
and for Chicago City from [Weather Underground] (https://www.wunderground.com/history/daily/us/il/chicago/KMDW).

We collected flight datasets from the following sources: 
for Incheon International Airport from [Air Portal] (https://data.kma.go.kr/data/air/selectAmosRltmList.do?pgmNo=575&tabNo=1), 
for New York City and Chicago City  from [United States Department of Transport]  (https://www.transtats.bts.gov/tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r).

There are the dataset in Data folder.

**- Examples of dataset**
- Examples of Weather dataset (JFK)
![그림1](https://github.com/dxlabskku/flightdelay/assets/119174391/5078c22a-0311-4e03-8786-6fb122c1f2ad)

- Examples of Flight dataset (JFK)
![그림2](https://github.com/dxlabskku/flightdelay/assets/119174391/b0c76a09-463d-4b03-87d7-cbb1d1be72cb)

# Models

This image shows the entire process of the models. We used light models for predicting flight departure delay
`ML_ICN.py`, `ML_JFK.py`, `LSTM_ICN.py` and `LSTM_JFK.py` in `Model` folder are the codes for experiments.
![그림3](https://github.com/dxlabskku/flightdelay/assets/119174391/a9443c7e-3a63-40e0-ba9b-493b87790cd9)




# Result
Test results for all models
- ICN
![그림4](https://github.com/dxlabskku/flightdelay/assets/119174391/2353594d-4cd2-496a-a2d9-a0b459ed4449)

- JFK
![그림5](https://github.com/dxlabskku/flightdelay/assets/119174391/a849009f-d880-400b-bf22-10d7e898975f)

- MDW
![그림6](https://github.com/dxlabskku/flightdelay/assets/119174391/f8e2e2fe-3b7d-457e-aae8-4edada3df94f)

Feature Importance for three airports
- ICN
![그림7](https://github.com/dxlabskku/flightdelay/assets/119174391/351c65e4-1306-41c9-aa7f-08c169515501)

- JFK
![그림8](https://github.com/dxlabskku/flightdelay/assets/119174391/872a8cbe-ac6e-4a1b-817f-816831dc0bc0)

- MDW
![그림9](https://github.com/dxlabskku/flightdelay/assets/119174391/3cb7a300-090f-459d-8dd8-6d8070229dcf)




