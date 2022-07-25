import csv
import pandas as pd
import numpy as np
import argparse
import os

def clean_HNO(filepath):
    '''assign rpt correctly'''
    output = ''
    with open(filepath, 'rt', encoding='cp1252') as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            access_instant = row[0]
            access_time = row[2]
            metric_name = row[3]
            user_id = row[5]
            # workstation_id = row[6]
            patID = row[6]
            csnID = row[7].strip()
            mnemonic_ids = row[8]

            note_type = row[15]
            prov_type = row[18]
            rpt15 = row[22]
            rpt16 = row[23]
            rpt17 = row[24]
            if 'nurse' in rpt15.lower() or 'physician assistant' in rpt15.lower():
                if 'nurse' in rpt16.lower():
                    rpt = rpt17
                else:
                    rpt = rpt16
            else:
                rpt = rpt15
            
            rpt = rpt.replace(',', '') # remove commas from rpt

            output += str(access_instant) + ',' + str(access_time) + ',' + metric_name + ',' + \
                        note_type + ',' + str(patID) + ',' + str(csnID) + ',' + \
                        user_id + ',' + mnemonic_ids + ',' + prov_type + ',' + rpt + '\n'
    f.close()

    g = open(filepath[:-4]+'_processed.csv', 'w')
    g.write(output)
    g.close()

def remove_duplicates(filepath):
    ''' After concatenating raw_mnemonics, HNO, and LRP, removes duplicate rows by comparing access_instant
    '''
    access_instant = 'ACCESS_INSTANT'; access_time = 'ACCESS_TIME'; metric_name = 'METRIC_NAME'; 
    patID = 'PAT_ID'; mnemonic_ids = 'DATA_MNEMONIC_ID'; #workstation_id = 'WORKSTATION_ID';
    report_name = 'REPORT_NAME'; csnID = 'CSN'; user_id = 'USER_ID'; prov_type = 'PROV_TYPE'; prov_dept = 'PROV_DEPT'
    output = ''

    f = open(filepath, 'r', encoding='utf-8-sig')
    is_first_row = True
    for l in f:
        row = l.split(',')
        access_instant2 = row[0]
        access_time2 = row[1]
        metric_name2 = row[2]
        report_name2 = row[3]
        patID2 = row[4]
        csnID2 = row[5]
        # workstation_id2 = row[6]
        user_id2 = row[6]
        mnemonic_ids2 = row[7]
        prov_type2 = row[8]
        prov_dept2 = row[9]

        if is_first_row:
            is_first_row = False
            continue
        elif (access_instant2 == access_instant) & (metric_name == metric_name2):
            mnemonic_ids += ';' + mnemonic_ids2
            if mnemonic_ids2 == 'HNO':
                report_name = report_name2
                prov_type = prov_type2
                prov_dept = prov_dept2
            elif mnemonic_ids2 == 'LRP':
                if report_name2 != '':
                    report_name = report_name2
        else:
            output += ','.join([access_instant, access_time, metric_name, report_name, patID, csnID, \
                               # workstation_id, \
                               user_id, mnemonic_ids, prov_type, prov_dept]) + '\n'
            access_instant = access_instant2
            access_time = access_time2
            metric_name = metric_name2
            report_name = report_name2
            patID = patID2
            csnID = csnID2
            # workstation_id = workstation_id2
            user_id = user_id2
            mnemonic_ids = mnemonic_ids2
            prov_type = prov_type2
            prov_dept = prov_dept2
            # losing the last event in the file though
    f.close()
    g = open(filepath[:-9]+'.csv', 'w')
    g.write(output)
    g.close()

def single_user_pipeline(filepath):
    ''' filepath = path/to/access_log_raw_mnemonics.csv, assuming _HNO, _LRP in same directory
        output is access_log_complete.csv in same directory
    '''
    # extract appropriate note metadata
    clean_HNO(filepath[:-18]+'_hno.csv')

    # read raw access log and cleaned hno files
    df_unfilt = pd.read_csv(filepath, encoding='cp1252')
    df_LRP = pd.read_csv(filepath[:-18]+'_LRP_Reports_View.csv', encoding='cp1252')
    df_HNO = pd.read_csv(filepath[:-18]+'_hno_processed.csv', encoding='cp1252')

    # harmonize column names
    col_headers = ['ACCESS_INSTANT', 'ACCESS_TIME', 'METRIC_NAME', 'REPORT_NAME', \
                    'PAT_ID', 'CSN', 'USER_ID', 'DATA_MNEMONIC_ID']
    df_LRP = df_LRP[col_headers]
    df_LRP['PROV_TYPE'] = ''; df_LRP['PROV_DEPT'] = ''
    df_unfilt['REPORT_NAME'] = ''
    df_unfilt = df_unfilt[col_headers]
    df_unfilt['PROV_TYPE'] = ''; df_unfilt['PROV_DEPT'] = ''
    df_HNO.rename(columns={'Note_Type':'REPORT_NAME', 'rpt15':'PROV_DEPT'}, inplace=True)

    # concatenate together and save
    df_output = pd.concat([df_LRP, df_unfilt, df_HNO], axis=0)
    df_output.sort_values(by='ACCESS_INSTANT', ascending=True, inplace=True, kind='mergesort')
    df_output.to_csv(filepath[:-18]+'_complete_wdup.csv', index=False)

    # remove duplicates
    remove_duplicates(filepath[:-18]+'_complete_wdup.csv')

    # Delete temp files created by this pipeline
    os.remove(filepath[:-18]+'_hno_processed.csv')
    os.remove(filepath[:-18]+'_complete_wdup.csv')

if __name__ == '__main__':
    # Usage: $ python access_log_pipeline.py access_log_raw_mnemonics.csv
    #    or  $ python access_log_pipeline.py ./SSIS_output/
    # Merges access_low_raw_mnemonics, access_log_HNO, access_log_LRP to produce access_log_complete.csv
    # Depends on the access_log_LRP and HNO type files to have reproducible naming structure

    # Parse arguments
    parser = argparse.ArgumentParser(description='Merges access_low_raw_mnemonics, \
        access_log_HNO, access_log_LRP to produce access_log_complete.csv')
    parser.add_argument('filepath', help='input access_log_raw_mnemonics.csv filepath or directory containing subfolders w such')
    args = parser.parse_args()

    if os.path.isfile(args.filepath): 
        # if a single access_log file is supplied, run on a single user
        single_user_pipeline(args.filepath)
    else: 
        # run on all subdirectories in the provided path/to/folder
        subfolders = [x[0] for x in os.walk(args.filepath)][1:] # 0th item is the root directory
        for f in subfolders:
            print(f)
            single_user_pipeline(f+'/access_log_raw_mnemonics.csv')

