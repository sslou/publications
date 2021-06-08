import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    df_note = df_note[(df_note.dos_time > timestamp - pd.DateOffset(window)) & (df_note.dos_time < timestamp)]

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


def measure_time(df, threshold = 300):
    ''' input:  df: access_log_complete type dataframe to analyze
                threshold: timeout interval in seconds, assumes user left computer 
        output: total_ehr_time: total time spent on EHR in hours
                afterhours_ehr_time: total time in hours spent on EHR btwn hours of 1800 and 0600
                review_time: total time in hours spent reviewing notes, results etc
                order_time: total time in hours reviewing and modifying orders
    '''
    # calculate total ehr time
    df['time_delta'] = df['ACCESS_INSTANT'].diff(periods=-1)*-1
    # set to NA where time delta is greater than some threshold
    df['time_delta'] = np.where((df.time_delta > threshold), np.nan, df.time_delta)
    total_ehr_time = df.time_delta.sum()/3600

    # calculate after hours time
    df_aft = df[(df.ACCESS_TIME.dt.hour > 18) | (df.ACCESS_TIME.dt.hour < 6)]
    afterhours_ehr_time = df_aft.time_delta.sum()/3600

    # calculate time spent on various categories
    cat_map = pd.read_csv('/storage1/lu/Active/ehrlog/ehr-logs/metric_categorized.csv')
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


def single_user_measure(filedir, timestamp, window=30):
    ''' input:  filedir: directory containing the access_log_complete.csv file
                timestamp: time of survey completion
                window: number of days prior to timestamp to evaluate
        returns list of measured parameters
    '''
    df = pd.read_csv(filedir + '/access_log_complete.csv')
    df.dropna(subset=['WORKSTATION_ID'], inplace=True)
    df.ACCESS_TIME = df.ACCESS_TIME.astype('datetime64')

    # select the time window of interest btwn timestamp and timestamp - window
    df = df[(df.ACCESS_TIME > timestamp - pd.DateOffset(window)) & (df.ACCESS_TIME < timestamp)]

    # calculate various statistics
    num_pts_viewed = df.PAT_ID.nunique()
    num_actions = df.shape[0]
    num_pts_per_day, num_pts_w_orders, num_pts_w_notes, num_days_w_orders, num_days_w_notes = measure_patients(df)
    num_orders = df[df.METRIC_NAME == 'Order list changed'].shape[0] # hope this is catching everything?
    num_messages = df[df.METRIC_NAME == 'In Basket message viewed'].shape[0]
    total_ehr_time, afterhours_ehr_time, review_time, order_time, inbox_time = measure_time(df)

    # measure note-related statistics
    df_note = pd.read_csv(filedir + '/note_writing_time_spent.csv')
    df_hno = pd.read_csv(filedir + '/access_log_hno.csv', encoding='cp1252')
    num_note, time_note, time_prog = measure_note(df_note, df_hno, timestamp, window)
    
    return [num_pts_viewed, num_actions, num_messages, num_orders, num_note, total_ehr_time, afterhours_ehr_time, 
            time_note, time_prog, review_time, order_time, inbox_time,
            num_pts_per_day, num_pts_w_orders, num_pts_w_notes, num_days_w_orders, num_days_w_notes]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='measures EHR usage characteristics based on access log data')
    parser.add_argument('--filepath', help='directory containing subfolders w access log data', default='none')
    args = parser.parse_args()

    # Specify where the access log data is
    if args.filepath == "none":
        root_dir = '/storage1/lu/Active/ehrlog/IGNITE/data_output_from_SSIS_2/'
    else:
        root_dir = args.filepath

    # Load survey data
    survey = pd.read_csv('/storage1/lu/Active/ehrlog/IGNITE/survey_data/survey_data_clean.csv')
    mapping = pd.read_csv('/storage1/lu/Active/ehrlog/IGNITE/survey_data/participants.csv')
    survey = survey.merge(mapping, on='record_id', how='inner')
    survey.time = survey.time.astype('datetime64')

    # Iterate through each survey completed and compute statistics
    output = []
    for i in range(survey.shape[0]):
        row = survey.iloc[i, :]
        survey_time = row.time
        USER_ID = row.USER_ID
        if pd.isnull(survey_time): # survey was not completed
            next_time = survey.iloc[min(i+1, survey.shape[0]-1), :].time
            if ~pd.isnull(next_time) and row.survey_num != 6:
                survey_time = next_time - pd.DateOffset(28)
                output_list = single_user_measure(root_dir + '/' + USER_ID, survey_time, window=30)
            else:
                output_list = [np.nan, np.nan]
        else:
            output_list = single_user_measure(root_dir + '/' + USER_ID, survey_time, window=30)
        output.append([USER_ID, survey_time] + output_list)

    result = pd.DataFrame(output, columns=['USER_ID', 'survey_time', 'num_pts_viewed', 'num_actions', 'num_messages',
                                            'num_orders', 'num_note', 'total_ehr_time', 'afterhours_ehr_time',
                                            'time_note', 'time_prog', 'review_time', 'order_time', 'inbox_time',
                                            'num_pts_per_day', 'num_pts_w_orders', 'num_pts_w_notes', 
                                            'num_days_w_orders', 'num_days_w_notes'])
    
    # Compute normalized measurements
    #   - num_pts_max: trying to count # patients user is responsible for as the larger of the # pts w orders/notes
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

    result.to_csv('./result/user_measurements.csv', index=False)
