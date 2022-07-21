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


def obtain_logs(dir):
    file = os.path.join(data_dir, dir, access_file_name)
    logs = pd.read_csv(file)
    # set dtype as "str" for digit USER_IDs
    logs['USER_ID'] = logs['USER_ID'].astype(str)

    # add session (chunk) index
    logs['action'] = logs['METRIC_NAME'].map(str) + '-' + logs['REPORT_NAME'].map(str)
    # logs['action'] = logs['METRIC_NAME'].map(str)
    logs['interval'] = logs['ACCESS_INSTANT'].diff(periods=1).fillna(value=0)
    chunk_starts = logs[logs['interval'] >= SESSION_BREAK * 60].index.tolist()
    start = 0
    for ind, end in enumerate(chunk_starts):
        logs.loc[np.logical_and(logs.index >= start, logs.index < end), 'chunk'] = ind
        start = end
    logs.loc[logs.index >= end, 'chunk'] = ind + 1
    logs['USER_ID'] = str(dir)
    logs['time_of_day'] = (pd.to_datetime(logs['ACCESS_TIME']) - pd.to_datetime(logs['ACCESS_TIME'])
                           .dt.normalize()).dt.total_seconds().div(3600).astype(float)
    
    logs_slice = logs[column_names]
    logs_slice.iloc[:]['USER_ID'] = logs_slice['USER_ID'].astype(str)
    logs_slice.iloc[:][['survey', 'shift', 'chunk', 'days_to_survey']] = logs_slice[
        ['survey', 'shift', 'chunk', 'days_to_survey']].astype(int)
    logs_to_append = logs_slice  # saving the whole data sequence

    return logs_to_append

# directories
data_dir = '../../IGNITE/data_output_from_SSIS_2'
survey_file = '../../IGNITE/survey_data/survey_data_clean_v2.csv'
participant_file = '../../IGNITE/survey_data/participants.csv'
access_file_name = 'access_log_complete.csv'
corpus_file = "../aux_data/corpus.pkl"
logs_file = '../aux_data/logs_all.csv'
dict_file = '../aux_data/token_dict.pkl'
survey_processed_file = '../aux_data/survey_processed.csv'

# params
SESSION_BREAK = 5  # min, action gap longer than this considered as a new session

# parsing from data files
data_dir = './IGNITE/data_output_from_SSIS_2/'
column_names = ['USER_ID', 'chunk', 'interval', 'ACCESS_TIME', 'time_of_day',
                    'action', 'PAT_ID']
num_cores = multiprocessing.cpu_count()
print('Parsing files...')
dir_list = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
log_list = Parallel(n_jobs=num_cores)(delayed(obtain_logs)(dir) for dir in tqdm(dir_list))
logs_all = pd.concat(log_list, axis=0)

# replacing action names to token ids
action_list = logs_all['action'].unique().tolist()
action_dict = dict(zip(action_list, map(str, np.array(range(len(action_list) + 1))[1:])))
inv_action_dict = {v: k for k, v in action_dict.items()}
logs_all['action_ID'] = logs_all['action'].apply(lambda x: action_dict[x])
logs_all.loc[:, 'USER_ID'] = logs_all['USER_ID'].astype(str)

filter_list=['USER_ID', 'chunk', 'ACCESS_TIME', 'action', 'action_ID']
filt_logs=logs_all[filter_list]
filt_logs.to_csv(logs_file, index=False)
print('Logs saved.')

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
