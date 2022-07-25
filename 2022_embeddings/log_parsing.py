# parsing access logs
import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import json
import pickle
import multiprocessing
from joblib import Parallel, delayed
from multiprocessing import Pool
import datetime


def list_folders(root_dir):
    ''' Use if root_dir contains many folders, one for each USER_ID
        i.e. root_dir/M####/access_log_complete.csv
    '''
    dir_list = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    return dir_list

def list_nested_paths(root_dir):
    ''' Use if there are multiple nested folders with data
    '''
    outer_list = list_folders(root_dir)
    nested_dir_list = []
    for d in outer_list:
        full_path = os.path.join(root_dir, d, 'provider_access_logs')
        inner_list = list_folders(full_path) # list of M###
        nested_dir_list += [os.path.join(d, 'provider_access_logs', f) for f in inner_list]
    return nested_dir_list

def obtain_logs(dir):
    file = os.path.join(data_dir, dir, access_file_name)
    try:
        logs = pd.read_csv(file)
    except:
        print(file + ' does not exist')
    
    # set dtype as "str" for consistency (there are some digit USER_IDs)
    logs['USER_ID'] = logs['USER_ID'].astype(str)

    # identify session ends
    logs['action'] = logs['METRIC_NAME'].map(str) + '-' + logs['REPORT_NAME'].map(str)
    logs['interval'] = logs['ACCESS_INSTANT'].diff(periods=-1).fillna(value=0)*-1
    chunk_ends = logs[logs['interval'] >= SESSION_BREAK * 60].index.tolist()
 
    # enumerate session IDs and label actions with their session ID 
    start = 0
    for ind, end in enumerate(chunk_ends):
        logs.loc[np.logical_and(logs.index >= start, logs.index < end+1), 'chunk'] = ind
        start = end + 1
    logs.loc[logs.index >= end, 'chunk'] = ind + 1

    return logs[column_names]

# directories
data_dir = '../../IGNITE/data_output_from_SSIS_2/'
access_file_name = 'access_log_complete.csv'
dir_list = list_folders(data_dir)
# dir_list = list_nested_paths(data_dir)

# output files
output_dir = '../aux_data/'
corpus_file = output_dir + 'corpus.pkl'
logs_file =  output_dir + 'logs_all.csv'
dict_file =  output_dir + 'token_dict.pkl'

# params
SESSION_BREAK = 5  # in minutes, action gap longer than this considered as a new session

# parsing from data files
column_names = ['USER_ID', 'chunk', 'interval', 'ACCESS_TIME', 'action', 'PAT_ID']
num_cores = multiprocessing.cpu_count()
print('Parsing files...')

log_list = Parallel(n_jobs=num_cores)(delayed(obtain_logs)(dir) for dir in tqdm(dir_list))
logs_all = pd.concat(log_list, axis=0)

# replacing action names to token ids
action_list = logs_all['action'].unique().tolist()
action_dict = dict(zip(action_list, map(str, np.array(range(len(action_list) + 1))[1:])))
inv_action_dict = {v: k for k, v in action_dict.items()}
logs_all['action_ID'] = logs_all['action'].apply(lambda x: action_dict[x])

# computing summary statistics
s = logs_all.groupby(by=['action_ID','action'])['interval'].agg(['sum', 'count']
    ).sort_values('count', ascending=False).reset_index()
s['sum'] = s['sum']/3600 # convert s to hrs
s.rename(columns={'sum':'total_hours', 'action_ID':'token'}, inplace=True)
# s['METRIC_NAME'] = s['action'].apply(lambda x: x.split('-')[0])
s.to_csv('../result/action_summary.csv', index=False)

# save logs (used for metric_only re-tokenization)
# filter_list=['USER_ID', 'chunk', 'ACCESS_TIME', 'action', 'action_ID']
# filt_logs=logs_all[filter_list]
# filt_logs.to_csv(logs_file, index=False) # save logs for tokenize_metric_name.py
# print('Logs saved.')

# save tokenized corpus for action embedding pretraining
print('Generating corpus for language model...')
corpus = []
for user in tqdm(logs_all['USER_ID'].unique()):
    logs = logs_all[logs_all['USER_ID'] == user]
    for chunk in logs['chunk'].unique():
        corpus.append(logs[logs['chunk'] == chunk]['action_ID'].astype('str').values.tolist())

with open(corpus_file, 'wb') as f:
    pickle.dump(corpus, f)
with open(dict_file, 'wb') as f:
    pickle.dump(inv_action_dict, f)
print('Corpus saved.')
