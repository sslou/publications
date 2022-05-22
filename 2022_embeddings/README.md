## Training Contextual Action Embeddings from Raw Audit Log Data

Code for Lou et al., Characterizing the microstructure of EHR work using raw audit logs: an unsupervised action embeddings approach

#### Parsing raw audit log files to generate the token list (vocabulary of actions)
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
                              --context 10 \ # number of contextual actions for embedding training
                              --if_train True \ # can change to "False" when trained embedding is saved
                              --if_metric_alone False \ # change to "True" if training action defined with metric_name alone
```
3. Word vectors are saved as `action_vectors_100.wv`

#### Embedding analysis and visualization using functions in `train_action_embedding.py`
1. t-SNE plotting of embedding vectors
```
tsne_out = compute_tsne(wv, df_metric_cat, metric_dict, token_dict,
                        perplexity=30, n_iter=5000, all_words=False)
plot_tsne(tsne_out, plot_nan = False, plot_ticks = False, all_labels=False)
tsne_out.to_csv("./result/tsne_out.csv", index=False)
```
2. Identifying actions in each cluster
```
identify_clusters("x1 > 10 and x2 > 5", tsne_out)
```
3. Find nearest neighbors for a given action, i.e. actions performed in similar contexts
```
action_name = 'Inpatient system list accessed-nan'
get_neighbors(action_name, wv, token_dict)
compute_topN_proximity(wv, token_dict, topN = 10)
```
