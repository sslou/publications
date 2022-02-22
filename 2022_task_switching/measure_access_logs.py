#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:21:41 2021

@author: Nigel Kim, Sunny Lou, Ben Warner

This file produces a descriptive table of basic measurements on a single day level (ICU attending by day - Physicians and Anesthesiologists)
"""
import pandas as pd
import numpy as np
import argparse
import tqdm
import os
import datetime
from collections import defaultdict

SESSION_INTERVAL = 5 * 60

def measure_note_subtypes(df_note, df_hno):
    ''' can be used to measure statistics on the number of times any given note was edited
        and temporal distrib of those edits, not yet implemented fully
    '''
    # left join to HNO to get note access times and PAT_IDs
    df_hno.ACCESS_TIME = df_hno.ACCESS_TIME.astype('datetime64')
    df_hno = df_hno[['ACCESS_INSTANT', 'ACCESS_TIME', 'METRIC_NAME', 'PAT_ID', 'NOTE_ID']]
    df_exp = df_note.merge(df_hno, how='left', on='NOTE_ID').sort_values(by=['DATE_OF_SERVICE_DTTM', 'ACCESS_INSTANT'])

    # filter only the notes the user worked on (vs viewed)
    df_exp = df_exp[(df_exp.METRIC_NAME == 'Clinical Note Signed') | 
                        (df_exp.METRIC_NAME == 'Pend clinical note') |
                        (df_exp.METRIC_NAME == 'Save clinical note')]

    # aggregate across NOTE_IDs to find first touch date, last touch date, and total touches for each note
    b = df_exp.groupby(['NOTE_ID', 'Note_Type', 'DATE_OF_SERVICE_DTTM', 'TOTAL_TIME_SECONDS'
                                ]).ACCESS_TIME.agg({'min', 'max', 'count'}).reset_index()
    b['dos_diff'] = (b['max'] - b['DATE_OF_SERVICE_DTTM'])/np.timedelta64(1, 's')/3600
    b['start_diff'] = (b['max'] - b['min'])/np.timedelta64(1, 's')/3600
    b['TOTAL_TIME_HOURS'] = b['TOTAL_TIME_SECONDS']/3600

    # if you wanted to measure number of notes with last touch past DOS date, would do it with b here

    # aggregate across all note types to find mean time spent on each type etc
    c = b.groupby('Note_Type')[['count', 'dos_diff', 'start_diff', 'TOTAL_TIME_HOURS']].agg({'mean', 'count'})
    c.columns = c.columns.map('_'.join)
    c = c[['count_count', 'count_mean', 
            'dos_diff_mean', 'start_diff_mean', 'TOTAL_TIME_HOURS_mean']]
    c.rename(columns={'count_count':'count', 'count_mean':'num_access_mean'}, inplace=True)
    try:
        hp_ct, hp_access, hp_dos_diff, hp_start_diff, hp_mean_time = c.loc['H&P']
    except:
        hp_ct, hp_access, hp_dos_diff, hp_start_diff, hp_mean_time = [0, np.nan, np.nan, np.nan, np.nan]

    # report 
    return hp_ct, hp_access, hp_mean_time


def measure_note(df_note, df_hno, timestamp, window):
    ''' input: 
            - df_note: dataframe resulting from loading note_writing_time_spent.csv
            - df_hno: df from access_log_HNO.csv
            - timestamp, window: specify over what time period to compute statistics
        output:
            - num_note: number of notes written over the specified time window
            - time_note: total time in hours spent writing notes over the time window
            - time_prog: total time in hours spent writing progress notes / H&Ps / consults
    '''
    # filter to relevant time interval
    df_note['dos_time'] = df_note.DATE_OF_SERVICE_DTTM.astype('datetime64')
    df_note = df_note[(df_note.dos_time < timestamp + pd.DateOffset(window)) & (df_note.dos_time >= timestamp)]

    # measure for all notes
    num_note = df_note.shape[0]
    time_note = df_note.TOTAL_TIME_SECONDS.sum()/3600 # time in hours

    # measure for progress notes / H&P
    time_prog = df_note[(df_note.Note_Type == 'Progress Notes') | 
                    (df_note.Note_Type == 'H&P') | 
                    (df_note.Note_Type == 'Subjective & Objective') |
                    (df_note.Note_Type == 'Assessment & Plan Note') |
                    (df_note.Note_Type == 'Consults') |
                    (df_note.Note_Type == 'Anesthesia Preprocedure Evaluation') |
                    (df_note.Note_Type == 'ED Provider Notes')
                    ].TOTAL_TIME_SECONDS.sum()/3600
    
    # measure_note_subtypes(df_note, df_hno) # TODO not finished yet

    return num_note, time_note, time_prog


def measure_patients(df):
    ''' input:  df: access_log_complete.csv type dataframe to analyze
        output: num_pts: number of unique patients per day with charts accessed in some way
                num_pts_w_orders/notes: on days with orders/notes, avg number of unique patients w orders/notes
                num_days_w_orders/notes: number of days with at least one order/note
    '''
    # calculate raw # of unique patients accessed per day
    num_pts_per_day = df.groupby(df.ACCESS_TIME.dt.date).PAT_ID.nunique().mean()
    # filter only for orders placed
    d = df[df.METRIC_NAME == 'Order list changed']
    num_pts_w_orders = d.groupby(d.ACCESS_TIME.dt.date).PAT_ID.nunique().mean()
    num_days_w_orders = d.groupby(d.ACCESS_TIME.dt.date).PAT_ID.count().shape[0]
    
    # filter only for notes signed
    d = df[df.METRIC_NAME == 'Clinical Note Signed']
    num_pts_w_notes = d.groupby(d.ACCESS_TIME.dt.date).PAT_ID.nunique().mean()
    num_days_w_notes = d.groupby(d.ACCESS_TIME.dt.date).PAT_ID.count().shape[0]
    return num_pts_per_day, num_pts_w_orders, num_pts_w_notes, num_days_w_orders, num_days_w_notes


def measure_time(df, which_shift=None):
    ''' input:  df: access_log_complete type dataframe to analyze
                which_shift: default=None. Specify 'AM' or 'PM'
        output: total_ehr_time: total time spent on EHR in hours
                afterhours_ehr_time: total time in hours spent on EHR btwn hours of 1800 and 0600
                review_time: total time in hours spent reviewing notes, results etc
                order_time: total time in hours reviewing and modifying orders
    '''
    # calculate total ehr time
    df['time_delta'] = df['ACCESS_INSTANT'].diff(periods=-1)*-1 # this modifies df in place!!
    # set to NA where time delta is greater than some threshold
    df['time_delta'] = np.where((df.time_delta > SESSION_INTERVAL), np.nan, df.time_delta)
    total_ehr_time = df.time_delta.sum()/3600
    # print(total_ehr_time*60)

    # calculate after hours time
    ## for AM shift: we captured logs between 12am-12am. therefore afterhours are time outside of 6am-6pm window
    ## for PM shift: we captured logs between 12pm today-12pm next day. 
    ##               therefore afterhours are times outside of 6pm-6am window in the log catchment window.
    if which_shift == 'AM':
        df_aft = df[(df.ACCESS_TIME.dt.hour > 18) | (df.ACCESS_TIME.dt.hour < 6)]
    elif which_shift == 'PM':
        df_aft = df[(df.ACCESS_TIME.dt.hour < 18) | (df.ACCESS_TIME.dt.hour > 6)]
    afterhours_ehr_time = df_aft.time_delta.sum()/3600

    # calculate time spent on various categories
    cat_map = pd.read_csv(os.path.join(os.path.curdir, 'aux_data', "metric_categorized.csv"))
    df = df.merge(cat_map, on=['METRIC_NAME', 'REPORT_NAME'], how='left')
    time_per_cat = df.groupby('metric_category').time_delta.sum() / 3600 # time per cat in hours
    try:
        chart_time = time_per_cat['Chart Review']
    except:
        chart_time = 0
    try:
        note_time = time_per_cat['Note Review']
    except:
        note_time = 0
    try:
        result_time = time_per_cat['Results Review']
    except:
        result_time = 0
    try:
        order_time = time_per_cat['Order Entry']
    except:
        order_time = 0
    try:
        inbox_time = time_per_cat['Inbox']
    except:
        inbox_time = 0
    review_time = chart_time + result_time + note_time
    return total_ehr_time, afterhours_ehr_time, review_time, order_time, inbox_time

def measure_switches(df, attending_date, window=1, which_shift=None, clean=True):
    ''' input:  
                df: access_log_complete file
                attending_date: attending timestamp
                window: default=1day
                which_shift: default=None. Specify 'AM' or 'PM'
                clean: default=True. specify whether the input dataframe is already cleaned
        returns:
                tot_switches: total number of switches
                avg_switches: mean average of 'switches per attending day'
                avg_sessions: mean average of 'sessions per attending day'
                avg_switch_per_session: mean avg of 'switches per session'
    '''
    if not clean:
        df = df[df.WORKSTATION_ID != "HAIKU"] # Get rid of irrelevant data.
        df = df[~df.WORKSTATION_ID.isnull()]
        df.reset_index(drop=True, inplace=True)
    
    # SESSION_INTERVAL = 5 * 60
    # switches = [np.nan]
    # sessions = [np.nan]
    # switch_per_session_list = [np.nan]
    
    # select the time window of interest btwn timestamp and timestamp - window -- Modified to reflect AM vs PM shifts
    if which_shift=='AM':
        sub_df = df[(df.ACCESS_TIME < attending_date + pd.DateOffset(window)) & (df.ACCESS_TIME >= attending_date)]
    elif which_shift=='PM':
        sub_df = df[(df.ACCESS_TIME >= attending_date + pd.DateOffset(hours=12)) & (df.ACCESS_TIME < attending_date + pd.DateOffset(hours=36))]
    else:
        sub_df = df[(df.ACCESS_TIME < attending_date + pd.DateOffset(window)) & (df.ACCESS_TIME >= attending_date)]
        
    # reset time deltas
    time_deltas = sub_df.loc[:, 'ACCESS_INSTANT'].diff(periods=-1)*-1
    sub_df['time_delta'] = np.where((time_deltas > SESSION_INTERVAL), np.nan, time_deltas)
    # sub_df.to_csv('./result/test.csv', index=False)
    
    cur_switches = 0
    cur_sessions = 0 # Session: Defined to be any period of time where there is an interval gap of less than five minutes.
    cur_pats_per_session = [] # Counts for unique patients encountered in each session
    cur_pat_lens = [] # Length of time spent for each switch.
    cur_pat_times = defaultdict(np.double) # amount of time per pat in this sub_df
    cur_sessions_duration_minute = []
    session_duration_start = 0
    time_bins = dict((t,0) for t in list(range(0,4*24))) # count of switches for each 15min bin
    time_bins_actions = dict((t,0) for t in list(range(0,4*24))) # count of actions for each 15min bin
    
    # to reset each session
    prev_pat = None 
    cur_pat_len = 0 # Length of time in a session.
    cur_patients = set() # Unique patients in a session.
    
    for idx, a_row in sub_df.iterrows():
        # For each action, who's the patient we're currently visiting?
        # LOOK AT EMPTY PATIENT ROWs
        # Filter out access log events without a patient ID.
        # Session start?
        if pd.isna(a_row.time_delta): # end of session
            if session_duration_start != 0:
                cur_sessions += 1
                # cur_sessions_duration_minute.append((a_row['ACCESS_TIME'] - session_duration_start)/np.timedelta64(1, 'm'))
                cur_sessions_duration_minute.append((a_row['ACCESS_INSTANT'] - session_duration_start)/60) # current: hour. not minute
                # print(a_row['ACCESS_INSTANT'])
                # print('session_duration: '+str((a_row['ACCESS_TIME'] - session_duration_start)/np.timedelta64(1, 'h')))
            if len(cur_patients) > 0: # only saves if user looked at least one patient this session
                cur_pats_per_session.append(len(cur_patients))
                cur_pat_lens.append(cur_pat_len)
            # reset
            prev_pat = None
            cur_patients = set()
            cur_pat_len = 0
            session_duration_start = 0 # placeholder value that represents None
        else:
            cur_pat = a_row.PAT_ID
            if pd.isna(cur_pat): # ignore lines where cur_pat is null
                continue
            elif cur_pat != prev_pat:# either start of session or switch
                if prev_pat is None: # start of session
                    prev_pat = cur_pat
                    cur_patients.add(cur_pat)
                    session_duration_start = a_row['ACCESS_INSTANT']
                    # print(a_row['ACCESS_INSTANT'])
                    # print(session_duration_start)
                else: #switch
                    if session_duration_start != 0:
                        cur_switches += 1
                        cur_patients.add(cur_pat)
                        cur_pat_lens.append(cur_pat_len)
                        # print(a_row)
                        if which_shift == 'AM':
                            date = datetime.date(1, 1, 1)
                            datetime1 = datetime.datetime.combine(date, a_row['ACCESS_TIME'].time())
                            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
                            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
                            time_bins[bin_num] += 1
                        elif which_shift =='PM':
                            date = datetime.date(1, 1, 1)
                            datetime1 = datetime.datetime.combine(date, (a_row['ACCESS_TIME']+datetime.timedelta(hours=12)).time())
                            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
                            # print(datetime1)
                            # print(datetime2)
                            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
                            time_bins[bin_num] += 1
                        else:
                            date = datetime.date(1, 1, 1)
                            datetime1 = datetime.datetime.combine(date, a_row['ACCESS_TIME'].time())
                            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
                            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
                            time_bins[bin_num] += 1
                        # reset
                        cur_pat_len = 0
                        prev_pat = cur_pat
            # increment
            cur_pat_len += a_row.time_delta
            cur_pat_times[cur_pat] += a_row.time_delta
            
        # record number of actions per time bin
        if which_shift == 'AM':
            date = datetime.date(1, 1, 1)
            datetime1 = datetime.datetime.combine(date, a_row['ACCESS_TIME'].time())
            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
            time_bins_actions[bin_num] += 1
        elif which_shift =='PM':
            date = datetime.date(1, 1, 1)
            datetime1 = datetime.datetime.combine(date, (a_row['ACCESS_TIME']+datetime.timedelta(hours=12)).time())
            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
            # print(datetime1)
            # print(datetime2)
            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
            time_bins_actions[bin_num] += 1
        else:
            date = datetime.date(1, 1, 1)
            datetime1 = datetime.datetime.combine(date, a_row['ACCESS_TIME'].time())
            datetime2 = datetime.datetime.combine(date, datetime.time(0,0,0))
            bin_num = int((datetime1 - datetime2) / datetime.timedelta(minutes=15))
            time_bins_actions[bin_num] += 1
            
            
    if cur_sessions == 0:
        switch_per_session = None
    else:
        switch_per_session = cur_switches/cur_sessions

    tot_switches = cur_switches
    tot_sessions = cur_sessions
    avg_switch_per_session = switch_per_session
    
    if cur_sessions == 0:
        avg_session_duration = None
    else:
        avg_session_duration = np.mean(cur_sessions_duration_minute)
        
    if cur_switches > 1:
        avg_timedelta_switches = np.mean(cur_pat_lens) # this would be in 'seconds'
    else:
        avg_timedelta_switches = None
    
    if (cur_sessions>0) & (cur_switches>0):
        switches_per_minute_session = tot_switches/np.sum(cur_sessions_duration_minute)
        # print(np.sum(cur_sessions_duration_minute))
    else:
        switches_per_minute_session = None
    
    num_switches_timeline = time_bins
    
    num_actions_timeline = time_bins_actions
    
    return (tot_switches, tot_sessions, avg_switch_per_session, avg_session_duration, avg_timedelta_switches, switches_per_minute_session, num_switches_timeline, num_actions_timeline)#, cur_pat_mean)

def single_user_measure(filedir, timestamp, window=1, shift=None):
    ''' input:  filedir: directory containing the access_log_complete.csv file
                timestamp: time of survey completion
                window: number of days prior to timestamp to evaluate
                shift: attending shift. either 'AM' or 'PM' or None (for NP and PAs)
        returns list of measured parameters
        
        Modified by Nigel: default window size = 1, window calculation changed to reflect AM and PM shifts.
                            AM shift: grab logs between 12am-12am
                            PM shift: grab logs between 12pm-12pm
    '''
    df = pd.read_csv(filedir + '/access_log_complete.csv')
    df.dropna(subset=['WORKSTATION_ID'], inplace=True)
    df.ACCESS_TIME = df.ACCESS_TIME.astype('datetime64')
    # TODO: consider filtering out days with few activities as "days off"

    # select the time window of interest btwn timestamp and timestamp - window -- Modified to reflect AM vs PM shifts
    if shift=='AM':
        df = df[(df.ACCESS_TIME < timestamp + pd.DateOffset(window)) & (df.ACCESS_TIME >= timestamp)]
    elif shift=='PM':
        df = df[(df.ACCESS_TIME >= timestamp + pd.DateOffset(hours=12)) & (df.ACCESS_TIME < timestamp + pd.DateOffset(hours=36))]
    else:
        df = df[(df.ACCESS_TIME < timestamp + pd.DateOffset(window)) & (df.ACCESS_TIME >= timestamp)]

    # calculate various statistics
    num_pts_viewed = df.PAT_ID.nunique()
    num_actions = df.shape[0]
    num_pts_per_day, num_pts_w_orders, num_pts_w_notes, num_days_w_orders, num_days_w_notes = measure_patients(df)
    # print(num_pts_w_notes)
    num_orders = df[df.METRIC_NAME == 'Order list changed'].shape[0] # hope this is catching everything?
    num_messages = df[df.METRIC_NAME == 'In Basket message viewed'].shape[0]
    num_chats = df[df.METRIC_NAME.str.contains('Secure chat')].shape[0] ##### SL EDIT 02.18.2022
    total_ehr_time, afterhours_ehr_time, review_time, order_time, inbox_time = measure_time(df, which_shift=shift)
    
    ## removing dependency to shift_work_hours_analysis() function.
    ## rewriting function in a simpler form. -- Just calculating tot_switches value for now.
    if df.shape[0] > 0:
        tot_switches, tot_sessions, avg_switch_per_session, avg_session_duration, avg_timedelta_switches, switches_per_minute_session, num_switches_timeline, num_actions_timeline = measure_switches(df, timestamp, window, which_shift=shift, clean=False)
    else:
        tot_switches, tot_sessions, avg_switch_per_session, avg_session_duration, avg_timedelta_switches, switches_per_minute_session, num_switches_timeline, num_actions_timeline = (None, None, None, None, None, None, None, None)

    # Measure number of patient switches.

    # measure note-related statistics
    df_note = pd.read_csv(filedir + '/note_writing_time_spent.csv')
    df_hno = pd.read_csv(filedir + '/access_log_HNO.csv', encoding='cp1252')
    num_note, time_note, time_prog = measure_note(df_note, df_hno, timestamp, window)
    
    return [num_pts_viewed, num_actions, num_messages, num_chats, num_orders, num_note, total_ehr_time, afterhours_ehr_time,
            time_note, time_prog, review_time, order_time, inbox_time, num_pts_per_day, num_pts_w_orders, num_pts_w_notes,
            num_days_w_orders, num_days_w_notes,
            tot_switches, tot_sessions, avg_switch_per_session, avg_session_duration, avg_timedelta_switches, switches_per_minute_session, num_switches_timeline, num_actions_timeline]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='measures EHR usage characteristics based on access log data')
    parser.add_argument('--filepath', help='directory containing subfolders w access log data', default='none')
    args = parser.parse_args()
    
    # Specify provider type (uncomment one)
    # clinician = 'physicians_anesthesiologists' #can run on local machine or SEAS computing cluster (must change file path)
    clinician = 'NP_and_PA' #need to run on SEAS computing cluster
    
    # Specify where the access log data is
    if args.filepath == "none":
        if clinician == 'physicians_anesthesiologists':
            # root_dir = os.path.join("/Volumes/Active/ehrlog", "icu_ehr_logs", "raw_data", "2019")
            root_dir = os.path.join("/storage1/lu/Active/ehrlog", "icu_ehr_logs", "raw_data", "2019") # for SEAS cluster
        else:
            root_dir = os.path.join("/storage1/lu/Active/ehrlog", "icu_ehr_logs", "raw_data", "2019")
    else:
        root_dir = args.filepath

    # Load survey data
    if args.filepath == 'none':
        
        if clinician == 'physicians_anesthesiologists': # ICU-attending physicians & anesthesiologists combined
            # attendings = pd.read_csv(os.path.join("/Volumes/Active/ehrlog", "icu_ehr_logs", "ICU_attending_"+clinician+"_2019.csv"))
            attendings = pd.read_csv(os.path.join("/storage1/lu/Active/ehrlog", "icu_ehr_logs", "ICU_attending_"+clinician+"_2019.csv"))# for SEAS cluster
        else: # ICD-shifts of nurse prcaticionters and physician assistants combined
            attendings = pd.read_csv(os.path.join("/storage1/lu/Active/ehrlog", "icu_ehr_logs", "ICU_shifts_"+clinician+"_2019.csv"))
    else:
        if clinician == 'physicians_anesthesiologists': # ICU-attending physicians & anesthesiologists combined
            attendings = pd.read_csv(os.path.join(args.filepath, '..', "ICU_attending_"+clinician+"_2019.csv"))
        else:# ICD-shifts of nurse prcaticionters and physician assistants combined
            attendings = pd.read_csv(os.path.join(args.filepath, '..', "ICU_shifts_"+clinician+"_2019.csv"))
    attendings['Assign Date'] = attendings['Assign Date'].astype('datetime64')
    
    # Uncomment this line to test the functions
    # attendings = attendings.head(1)
    # print(attendings['USER_ID'])
    
    # Iterate through each survey completed and compute statistics
    output = []
    pbar = tqdm.tqdm(total=attendings.shape[0], desc="Attendings Processed")
    for i in range(attendings.shape[0]):
        row = attendings.iloc[i, :]
        attending_time = row['Assign Date']
        USER_ID = row.USER_ID
        if clinician == 'physicians_anesthesiologists':
            if row['Shift Group']=='SCIU-PM Red':
                output_list = single_user_measure(root_dir + '/' + USER_ID, attending_time, window=1, shift='PM')
            else:
                output_list = single_user_measure(root_dir + '/' + USER_ID, attending_time, window=1, shift='AM')
        else:
            output_list = single_user_measure(root_dir + '/' + USER_ID, attending_time, window=1, shift='AM')
        output.append([USER_ID, attending_time] + output_list)
        # print(USER_ID)
        pbar.update(1)
        
    result = pd.DataFrame(output, columns=['USER_ID', 'attending_time', 'num_pts_viewed', 'num_actions', 'num_messages', 'num_chats',
                                            'num_orders', 'num_note', 'total_ehr_time', 'afterhours_ehr_time',
                                            'time_note', 'time_prog', 'review_time', 'order_time', 'inbox_time',
                                            'num_pts_per_day', 'num_pts_w_orders', 'num_pts_w_notes', 
                                            'num_days_w_orders', 'num_days_w_notes','tot_switches','tot_sessions',
                                            'avg_switch_per_session', 'avg_session_duration', 'avg_timedelta_switches','switches_per_minute_session',
                                            'num_switches_timeline','num_actions_timeline'])
    
    # Compute normalized measurements
    #   - num_pts_max: trying to count # patients user is responsible for as the larger of the # pts w orders/notes
    #       TODO: is there anything better for counting # pts directly responsible for?
    #   - norm_time_note: avg hours spent on notes per patient per day
    #   - norm_num_orders: avg # orders placed per patient per day
    #   - norm_actions: avg # actions per patient per working day
    #   - norm_ehr_time: avg hours using EHR per patient day
    #   - fraction_afterhours: fraction of total EHR time spent between 1800 and 0600
    #   - norm_review_time: avg hours reviewing data per patient day
    #   - fraction_review: fraction of total EHR time spent reviewing data
    #   - norm_order_time: avg seconds spent on orders per order
    #   - norm_inbox_time: avg seconds spent on inbox per inbox message reviewed
    num_days_working = result.num_days_w_notes 
    #       TODO: find a better way to measure # working days, currently using # days w notes
    result['num_pts_max'] = result[['num_pts_w_orders', 'num_pts_w_notes']].max(axis=1)
    result['norm_time_note'] = result.time_note / result.num_pts_w_notes / result.num_days_w_notes
    result['norm_num_orders'] = result.num_orders / result.num_days_w_orders / result.num_pts_w_orders
    result['norm_actions'] = result.num_actions / num_days_working / result.num_pts_max
    result['norm_ehr_time'] = result.total_ehr_time / num_days_working / result.num_pts_max
    result['fraction_afterhours'] = result.afterhours_ehr_time / result.total_ehr_time
    result['norm_review_time'] = result.review_time / num_days_working / result.num_pts_max
    result['fraction_review'] = result.review_time / result.total_ehr_time
    result['norm_order_time'] = result.order_time / result.num_orders * 3600
    result['norm_inbox_time'] = result.inbox_time / result.num_messages * 3600
    result['switch_rate'] = result.tot_switches / result.num_actions
    result['switch_rate_100'] = result.tot_switches / result.num_actions * 100
    
    # print(result['num_pts_w_notes'])
    if clinician == 'physicians_anesthesiologists':
        # result.to_csv('./result/user_measurements_ICUattendings_'+clinician+'_01242022_timecourse.csv', index=False)
        result.to_csv('/storage1/lu/Active/ehrlog/icu_ehr_logs/task_switching_Nigel/result/'+
                      'user_measurements_ICUattendings_'+clinician+'_02182022_timecourse.csv', index=False)
    else: # NP and PA
        result.to_csv('/storage1/lu/Active/ehrlog/icu_ehr_logs/task_switching_Nigel/result/'+
                      'user_measurements_NP_PA_shifts_02182022_timecourse.csv', index=False)
    
    # Must add the "Gender" column separately to the user measurements file using Jupyter notebook IGNITE_ICU_mapping_with_ICU_AttendingShift_List.ipynb