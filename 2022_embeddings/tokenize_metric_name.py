# parsing access logs
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


corpus_file = "../aux_data/corpus_metric_name.pkl"
logs_file = '../aux_data/logs_all.csv'
dict_file = '../aux_data/token_dict_metric_name.pkl'
logs_all = pd.read_csv(logs_file)

# replacing action names to token ids
logs_all['metric_name'] = logs_all['action'].apply(lambda x: x.split('-')[0])
logs_all['report_name'] = logs_all['action'].apply(lambda x: x.split('-')[1])

metric_list = logs_all['metric_name'].unique().tolist()
metric_dict = dict(zip(metric_list, map(str, np.array(range(len(metric_list) + 1))[1:])))
inv_metric_dict = {v: k for k, v in metric_dict.items()}
logs_all['metric_ID'] = logs_all['metric_name'].apply(lambda x: metric_dict[x])
logs_all.loc[:, 'USER_ID'] = logs_all['USER_ID'].astype(str)

logs_all.to_csv(logs_file, index=False)
print('Logs saved.')


# save tokenized corpus for action embedding pretraining
print('Generating corpus for language model...')
corpus = []
for user in tqdm(logs_all['USER_ID'].unique()):
    logs = logs_all[logs_all['USER_ID'] == user]
    for chunk in logs['chunk'].unique():
        corpus.append(logs[logs['chunk'] == chunk]['metric_ID'].astype('str').values.tolist())

with open(corpus_file, 'wb') as f:
    pickle.dump(corpus, f)
with open(dict_file, 'wb') as f:
    pickle.dump(inv_metric_dict, f)
print('Corpus saved.')

