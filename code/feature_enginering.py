# /usr/bin/python3

# il faut executer cette commande:
# !pip install meteocalc
# sur le notebook au debut.

import pandas as pd
import numpy as np
import datetime

from datetime import date
from meteocalc import feels_like, Temp



def fill_weather_dataset(weather_df):
  
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day 
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    weather_df = weather_df.set_index(['site_id','day','month'])
     
    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)
    
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])
    weather_df.update(cloud_coverage_filler,overwrite=False)
    
    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)  

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])
    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)


    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])
    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
    
    del air_temperature_filler, cloud_coverage_filler, sea_level_filler, precip_depth_filler 
    return weather_df




def calculate_rh(df):
    a,b = 17.625, 243.04
    df['relative_humidity'] = np.exp( (a * df['dew_temperature']) / (b + df['dew_temperature']))
    df['relative_humidity'] = 100 * df['relative_humidity'] / np.exp((a * df['air_temperature'])/(b + df['air_temperature']))
    return df

def calculate_fl(df):
  flike_final = []
  flike = []
  # calculate Feels Like temperature
  for i in range(len(df)):
      at = df['air_temperature'][i]
      rh = df['relative_humidity'][i]
      ws = df['wind_speed'][i]
      flike.append(feels_like(Temp(at, unit = 'C'), rh, ws))
  for i in range(len(flike)):
      flike_final.append(flike[i].f)
  df['feels_like'] = flike_final
  del flike_final, flike, at, rh, ws

def get_meteorological_features(data):
  calculate_rh(data)
  calculate_fl(data)
  return data


def time_features(df): 
  # Sort by timestamp
  df.sort_values("timestamp")
  df.reset_index(drop=True)
  
  # Add more features
  df["timestamp_2"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
  df["hour"] = df["timestamp_2"].dt.hour
  df["dayofweek"] = df["timestamp_2"].dt.weekday

  df['month_group'] = df['timestamp_2'].dt.month 
  df['month_group'].replace((1, 2), 1, inplace = True)
  df['month_group'].replace((3, 4, 5, 6, 7), 2, inplace = True)
  df['month_group'].replace((8, 9, 10, 11, 12), 3, inplace = True)

  df = df.drop('timestamp_2' , 1)

  return df


def is_holiday(df):
    ## https://www.geeksforgeeks.org/python-holidays-library/
    ## https://towardsdatascience.com/5-minute-guide-to-detecting-holidays-in-python-c270f8479387
    import holidays
    YEARS = (2016,2017)
    UK=[]
    for year in YEARS : 
        for ptr in holidays.UnitedKingdom(years=year).keys(): 
            UK.append(str(ptr))

    for ptr in holidays.UnitedKingdom(years=2018).keys():
        UK.append(str(ptr))
        UK.append('2019-01-01')
    

    IR=[]
    for year in YEARS : 
        for ptr in holidays.Ireland(years=year).keys():  #2016 year holydays in ireland
            IR.append(str(ptr))

    for ptr in holidays.Ireland(years=2018).keys():
        IR.append(str(ptr))
        IR.append('2019-01-01')
    

    US=[]
    for year in YEARS :
        for ptr in holidays.UnitedStates(years=year).keys(): #2016 year holydays in US
            US.append(str(ptr))

    for ptr in holidays.UnitedStates(years=2018).keys():
        US.append(str(ptr))
        US.append('2019-01-01')
    
    CA=[]
    for year in YEARS :
        for ptr in holidays.Canada(years=year).keys():   #2016 year holidays in Canada
            CA.append(str(ptr))
    for ptr in holidays.Canada(years=2018).keys():
        CA.append(str(ptr))
        CA.append('2019-01-01')

    location = pd.DataFrame()
    location['site_id'] = np.arange(0,16)


    location['city'] = ['Orlando','Heathrow','Tempe','Washington','Berkeley','Southampton',\
                        'Washington','Ottowa','Orlando','Austin','Saltlake','Ottowa','Dublin',\
                          'Minneapolis','Philadelphia','Rochester']
    location['country'] = ['US','UK','US','US','US','UK',\
                        'US','Montreal','US','US','US','Montreal','Ireland',\
                        'US','US','US']

    df = df.merge(location, on='site_id', how='left')
    df['is_holiday'] = [0]*(df.shape[0])
    timestamp = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
    df.loc[df['country']=='US','is_holiday'] = (timestamp.dt.date.astype('str').isin(US)).astype(int)
    df.loc[df['country']=='UK','is_holiday'] = (timestamp.dt.date.astype('str').isin(UK)).astype(int)
    df.loc[df['country']=='Montreal','is_holiday'] = (timestamp.dt.date.astype('str').isin(CA)).astype(int)
    df.loc[df['country']=='Ireland','is_holiday'] = (timestamp.dt.date.astype('str').isin(IR)).astype(int)
    return df



def prepare_weather_data(weather):
    weather = fill_weather_dataset(weather)
    weather = get_meteorological_features(weather)
    weather = is_holiday(weather)

    return weather


def prepare_building_data(building):

    building = building.fillna(value={'floor_count':0})
    building['year_built'] = building['year_built'].fillna(building['year_built'].mean())

    return building

