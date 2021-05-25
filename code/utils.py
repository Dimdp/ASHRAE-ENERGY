# /usr/bin/python3

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from collections import Counter

def reduce_mem_usage(df, use_float16=False, verbose=False):
    """
    Iterate through all the columns of a dataframe and modify the data type 
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose : 
    	print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    	print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df



class Data_Filler():
    def __init__(self , cols_to_fill_median, cols_to_fill_categ):
        self.cols_to_fill_median = cols_to_fill_median
        self.cols_to_fill_categ = cols_to_fill_categ
        self.fillers = {}

    def fit(self , data) : 
        for col in self.cols_to_fill_median:
            self.fillers[col] = data[col].median(axis=0,skipna=True)
        
        for col in self.cols_to_fill_categ:
            tmp_var = list(data[col].dropna().values)
            tmp_data = Counter(tmp_var)
            self.fillers[col] = max(tmp_var, key=tmp_data.get)

    def transform(self , data):
        for col in self.cols_to_fill_median:
            data[col].fillna(self.fillers[col] , inplace = True)
        
        for col in self.cols_to_fill_categ:
            data[col].fillna(self.fillers[col] , inplace = True)

        return data

class Data_Scaler_Encoder():
    def __init__(self , cols_to_scale , cols_to_drop , cols_to_encode):
        self.cols_to_scale = cols_to_scale
        self.cols_to_drop = cols_to_drop
        self.cols_to_encode = cols_to_encode
        self.stats = {}
        self.label_encoders = {}
        
    def fit(self , data):
        for col in self.cols_to_scale:
            mean_col = data[col].mean()
            std_col = data[col].std()
            self.stats[col] = (mean_col , std_col)

        for col in self.cols_to_encode:
            le = LabelEncoder()
            le.fit(data[col])
            self.label_encoders[col] = le
        
    def transform(self , data):
        data = data.drop(self.cols_to_drop , axis = 1)

        for col in self.cols_to_scale:
            data[col] = ( data[col] - self.stats[col][0])/ self.stats[col][1]

        for col in self.cols_to_encode:
            data[col] = self.label_encoders[col].transform(data[col])

        return data

