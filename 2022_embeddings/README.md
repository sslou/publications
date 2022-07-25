## Training Contextual Action Embeddings from Raw Audit Log Data

Code for Lou et al., Characterizing the macrostructure of EHR work using raw audit logs: an unsupervised action embeddings approach

#### Augmenting raw audit log data with additional report and note detail
1. Pull relevant audit log data and associated metadata from Clarity using `detailed_audit_log_query.sql`. This query generates 4 files for each user:  
    * `access_log_raw.csv` (i.e. the raw `ACCESS_LOG` table)
    * `access_log_raw_mnemonics.csv` (shows all mnemonic metadata available for each action in the raw audit log, i.e., there is one row for each action and mnemonic available)
    * `access_log_HNO.csv` (contains `REPORT_NAME` for only those audit log entries with HNO mnemonic metadata)
    * `access_log_LRP_Reports_View.csv` (contains `REPORT_NAME` for only those audit log entries with LRP mnemonic metadata)
        
2. Generate the detailed audit log by combining the 4 generated files above using `access_log_pipeline.py`.
    * output file: `access_log_complete.csv`, this is the final detailed audit log with `METRIC_NAME` and `REPORT_NAME` columns (where available)

#### Parsing detailed audit log files to generate the token list (vocabulary of actions)
1.  `log_parsing.py` - Reads all audit logs, tokenizes `METRIC_NAME`-`REPORT_NAME` pairs, splits audit logs into sessions, and writes the corpus of tokenized actions to disk.
2.  `tokenize_metric_name.py` - Tokenizes using `METRIC_NAME` alone, and writes the metric-only corpus to disk.
```
python log_parsing.py
python tokenize_metric_name.py
```
#### Using SkipGram (Word2Vec) to train contextual action embeddings
1. Install Word2Vec toolkit GenSim
```
pip install gensim==3.8.3
```
2. Train action embedding
```
python train_action_embedding --embedding_size 100 \
                              --epochs 20 \
                              --context 10 \ # window length for embedding training
                              --if_train True \ # can change to "False" when trained embedding is saved
                              --if_metric_alone False \ # change to "True" if training action defined with metric_name alone
```
3. Word vectors are saved in `./aux_data/action_vectors_100.wv`

#### Embedding analysis and visualization using functions in `train_action_embedding.py`
1. t-SNE plotting of embedding vectors
```
tsne_out = compute_tsne(wv, df_metric_cat, metric_dict, token_dict,
                        perplexity=30, n_iter=5000, all_words=False)
plot_tsne(tsne_out, plot_nan = False, plot_ticks = False, all_labels=False)
tsne_out.to_csv("./result/tsne_out.csv", index=False)
```
2. Identifying actions in each cluster. Clusters described in the manuscript are listed in `./tsne_groups/`
```
identify_clusters("x1 > 10 and x2 > 5", tsne_out)
```
3. Find nearest neighbors for a given action, i.e. actions performed in similar contexts
```
action_name = 'Inpatient system list accessed-nan'
get_neighbors(action_name, wv, token_dict)
compute_topN_proximity(wv, token_dict, topN = 10)
```
