import pandas as pd
from datetime import timedelta

def preprocess(data, day_window = 3, **kwargs):
    #cast day0date and act date to date
    data.loc[:,'day0_date'] = pd.to_datetime(data.loc[:,'day0_date'])
    data.loc[:,'act_date'] = pd.to_datetime(data.loc[:,'act_date'])
    if (data.act_date.max() - data.day0_date.min()) > timedelta(3):
        data = data[data.loc[:,'act_date'] < data.loc[:,'act_date'].max()] #drop max act_date 
        data = data[data.loc[:,'day0_date'] <= data.loc[:,'act_date'].max()-timedelta(days=day_window)] #find range day0_date
        data = data[(data.loc[:,'day_diff'] <= day_window) & (data.loc[:,'day_diff']>= 0)]

    for key, value in kwargs.items():
        data = data[data.loc[:,key] == value]
    
    data['combined_ad'] = data['rv_imp_sum'] * 1.3 + data['is_imp_sum']
    return data

def get_abtest_data(df, target, **kwargs):
    ab_data = preprocess(df, **kwargs)
    ab_data = ab_data.groupby('resettable_device_id_or_app_instance_id')[target].sum().reset_index(drop=True)
    return ab_data

def gameinfo(df, METRICS, **kwargs):
    data = preprocess(df, day_window = 3, **kwargs)
    data_by_day_diff = data.groupby('day_diff')[METRICS].sum()
    data_by_day_diff.loc['Grand Total'] = data_by_day_diff.iloc[:,1:].sum()

    data_per_dau = data_by_day_diff.copy()
    data_per_dau.iloc[:-1,1:] = data_per_dau.iloc[:-1,1:].div(data_per_dau.iloc[:-1,0], axis=0)
    data_per_dau.iloc[-1,1:] = data_per_dau.iloc[-1,1:]/data_per_dau.iloc[0,0]
    data_per_dau['dau'] = data_per_dau['dau']/data_per_dau.iloc[0,0]
    data_per_dau = data_per_dau.style.format({'dau': "{:.2%}"})
    return data_by_day_diff, data_per_dau

def maxstage(data,**kwargs):
    data = preprocess(data, **kwargs)
    max_stage = data.groupby('resettable_device_id_or_app_instance_id')['max_stage'].max().reset_index()
    max_stage = max_stage.groupby('max_stage').agg('count').rename({'resettable_device_id_or_app_instance_id':'dau'}, axis=1)[:16]
    max_stage.loc['Grand Total'] = max_stage[:16].sum()
    user_left = [max_stage.iloc[-1,:] - max_stage.iloc[0,0]]
    dau = pd.Series(max_stage.dau)
    for row in dau.index[1:-1]:
        user_left.append(user_left[row-1] - dau[row])
    user_left = pd.DataFrame(user_left).reset_index(drop=True)
    max_stage = pd.concat([max_stage, user_left], axis=1)
    max_stage.index.names = ['stage']
    max_stage.columns = ['dau','user_left']
    max_stage['drop_rate'] = (max_stage.user_left / max_stage.iloc[-1,0]).map("{:.2%}".format)
   
    return max_stage