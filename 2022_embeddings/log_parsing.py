# parsing access logs
import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import json
import pickle
from shift_analysis import ShiftEstimation
import multiprocessing
from joblib import Parallel, delayed
from multiprocessing import Pool
import datetime


def obtain_logs(dir):
    file = os.path.join(data_dir, dir, access_file_name)
    logs = pd.read_csv(file)
    # set dtype as "str" for digit USER_IDs
    logs['USER_ID'] = logs['USER_ID'].astype(str)

    # add shift index
    shift_estimator = ShiftEstimation(logs, T_basic=4, T_small_2=7, T_large_2=30, T_small_3=2, T_large_3=20)
    df_shift = shift_estimator.shift_estimation()
    for shift_ind in df_shift.index:
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= pd.to_datetime(df_shift.loc[shift_ind, 'start']),
                                pd.to_datetime(logs['ACCESS_TIME']) <= pd.to_datetime(df_shift.loc[shift_ind, 'end'])), 'shift'] = shift_ind

    # add survey index # not considering logs prior to this time
    first_survey = df_survey[np.logical_and(df_survey['USER_ID'] == dir, df_survey['survey_num'] == 1)].iloc[0]['time']
    last_survey_time = pd.to_datetime(first_survey) - datetime.timedelta(DATA_WINDOW)

    # add survey index 1-6
    for survey_ind in df_survey[df_survey['USER_ID'] == dir]['survey_num'].values:
        survey_time_str = df_survey[np.logical_and(df_survey['USER_ID'] == dir, df_survey['survey_num'] == survey_ind)].iloc[0]['time']
        survey_time = pd.to_datetime(survey_time_str)
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= last_survey_time,
                                pd.to_datetime(logs['ACCESS_TIME']) < survey_time), 'survey'] = survey_ind
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= last_survey_time,
                                pd.to_datetime(logs['ACCESS_TIME']) < survey_time), 'survey_time'] = survey_time
        last_survey_time = survey_time

    # mark survey 7, 8, ... for actions after last survey
    while len(logs.loc[pd.to_datetime(logs['ACCESS_TIME']) > last_survey_time]) > 0:
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= last_survey_time,
                                pd.to_datetime(logs['ACCESS_TIME']) < last_survey_time + datetime.timedelta(DATA_WINDOW)), 'survey'] = survey_ind + 1
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= last_survey_time,
                                pd.to_datetime(logs['ACCESS_TIME']) < last_survey_time + datetime.timedelta(DATA_WINDOW)), 'survey_time'] = last_survey_time + datetime.timedelta(DATA_WINDOW)
        last_survey_time = last_survey_time + datetime.timedelta(DATA_WINDOW)
        survey_ind = survey_ind + 1

    # mark survey 0, -1, -2 for actions over 1 month before first survey
    survey_time = pd.to_datetime(first_survey) - datetime.timedelta(DATA_WINDOW)
    survey_ind = 1
    while len(logs.loc[pd.to_datetime(logs['ACCESS_TIME']) < survey_time]) > 0:
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= survey_time - datetime.timedelta(DATA_WINDOW),
                                pd.to_datetime(logs['ACCESS_TIME']) < survey_time), 'survey'] = survey_ind - 1
        logs.loc[np.logical_and(pd.to_datetime(logs['ACCESS_TIME']) >= survey_time - datetime.timedelta(DATA_WINDOW),
                                pd.to_datetime(logs['ACCESS_TIME']) < survey_time), 'survey_time'] = survey_time
        survey_time = survey_time - datetime.timedelta(DATA_WINDOW)
        survey_ind = survey_ind - 1

    # add session (chunk) index
    logs['action'] = logs['METRIC_NAME'].map(str) + '-' + logs['REPORT_NAME'].map(str)
    logs['interval'] = logs['ACCESS_INSTANT'].diff(periods=1).fillna(value=0)
    chunk_starts = logs[logs['interval'] >= SESSION_BREAK * 60].index.tolist()
    start = 0
    for ind, end in enumerate(chunk_starts):
        logs.loc[np.logical_and(logs.index >= start, logs.index < end), 'chunk'] = ind
        start = end
    logs.loc[logs.index >= end, 'chunk'] = ind + 1
    logs['USER_ID'] = str(dir)
    logs['days_to_survey'] = (
            pd.to_datetime(logs['survey_time']).dt.normalize() - pd.to_datetime(logs['ACCESS_TIME'])
            .dt.normalize()).dt.total_seconds().div(3600 * 24)
    logs['time_of_day'] = (pd.to_datetime(logs['ACCESS_TIME']) - pd.to_datetime(logs['ACCESS_TIME'])
                           .dt.normalize()).dt.total_seconds().div(3600).astype(float)
    logs['survey'] = logs['survey'].fillna(value=-1)  # -1 marks no survey outcome available
    logs_slice = logs[column_names]
    logs_slice.iloc[:]['USER_ID'] = logs_slice['USER_ID'].astype(str)
    logs_slice.iloc[:]['days_to_survey'] = logs_slice['days_to_survey'].fillna(value=-1)  # -1 marks no survey outcome available
    logs_slice.iloc[:][['survey', 'shift', 'chunk', 'days_to_survey']] = logs_slice[
        ['survey', 'shift', 'chunk', 'days_to_survey']].astype(int)
    logs_to_append = logs_slice  # saving the whole data sequence

    return logs_to_append


def process_survey(survey_file, participant_file):
    # load files
    survey = pd.read_csv(survey_file)
    participant = pd.read_csv(participant_file)
    # add USER_ID from participant file
    participant.loc[:, 'USER_ID'] = participant['USER_ID'].astype(str)
    recordid2USER_ID = dict(zip(participant.record_id.values.tolist(), participant.USER_ID.values.tolist()))
    survey['USER_ID'] = survey['record_id'].transform(lambda x: recordid2USER_ID[x])
    survey['time'] = survey['time'].transform(lambda x: pd.to_datetime(x))
    # survey time imputation for months with no survey
    for user in survey['USER_ID'].unique():
        df = survey[survey['USER_ID'] == user]
        # add reference time (6 month later)
        df_ext = df.append({'time': pd.to_datetime(df[df.survey_num == 1].time.values[0]) + datetime.timedelta(6 * 30)},
                           ignore_index=True)
        # interpolate time
        datetimes_imputed = df_ext.time.transform(
            lambda x: (pd.to_datetime(x) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).interpolate().transform(
            lambda x: pd.to_datetime(x, unit='s')).iloc[:-1]
        # use interpolated time
        survey.loc[survey['USER_ID'] == user, 'time'] = datetimes_imputed.values
        # add sample_ID
        survey['sample_ID'] = survey['USER_ID'].map(str) + '-' + survey['survey_num'].map(str)

    return survey, recordid2USER_ID




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
SESSION_BREAK = 10  # min, action gap longer than this considered as a new session
DATA_WINDOW = 30  # days, time length of data to use for one survey

# load and process files
df_survey, recordid2USER_ID = process_survey(survey_file, participant_file)
df_survey.to_csv(survey_processed_file, index=False)

# parsing from data files
print('Parsing files...')
column_names = ['USER_ID', 'survey', 'shift', 'chunk', 'interval', 'ACCESS_TIME', 'days_to_survey', 'time_of_day',
                'action', 'PAT_ID']
num_cores = multiprocessing.cpu_count()
dir_list = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
log_list = Parallel(n_jobs=num_cores)(delayed(obtain_logs)(dir) for dir in tqdm(dir_list))
logs_all = pd.concat(log_list, axis=0)


# replacing action names to token ids
action_list = logs_all['action'].unique().tolist()
action_dict = dict(zip(action_list, map(str, np.array(range(len(action_list) + 1))[1:])))
inv_action_dict = {v: k for k, v in action_dict.items()}
logs_all['action_ID'] = logs_all['action'].apply(lambda x: action_dict[x])
logs_all.loc[:, 'USER_ID'] = logs_all['USER_ID'].astype(str)
# add sample_ID and sample_shift_ID
logs_all['sample_ID'] = logs_all['USER_ID'].map(str) + '-' + logs_all['survey'].map(int).map(str)
logs_all['sample_shift_ID'] = logs_all['USER_ID'].astype(str) + '_' + logs_all['shift'].astype(int).astype(str)
logs_all.to_csv(logs_file, index=False)
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


