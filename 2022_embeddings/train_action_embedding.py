import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors  # pip install gensim==3.8.3
from gensim.models.callbacks import CallbackAny2Vec
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
import sklearn.metrics


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class lossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        # print(f'  Loss: {loss}')
        self.epoch += 1


def train_embeddings(corpus, args):
    # train embeddings
    print('Training action embeddings ...')
    model = Word2Vec(sentences=corpus,
                     iter=args.epochs, # gensim 3.8.3
                     size=args.embedding_size,
                     # epochs = args.epochs, # gensim 4.2.0
                     # vector_size = args.embedding_size
                     window=args.context,
                     min_count=args.min_count,
                     workers=10,
                     callbacks=[lossLogger()],
                     sg=1  # sg=1 means using skipgram
                     )

    # save model
    print('Training complete!')
    wv = model.wv
    return wv


def compute_tsne(wv, df_metric_cat, metric_dict, token_dict, all_words=True, perplexity=40, n_iter=2000):
    ''' wv - gensim keyedvectors model
        df_metric_cat - pd array of vocabulary, counts, and category
        metric_dict - maps str action name to category
        token_dict - maps token number (as str) to str text of action name
        all_words - whether to include the entire vocabulary or some subset in tsne training

        returns pd DataFrame w columns: x1, x2, token, action, count, metric_category
    '''
    df_metric_cat.sort_values(by='count', ascending=False, inplace=True)
    if all_words:
        num_embedding = len(wv.index2word) # size of vocab
    else:
        # num_embedding = 1000 # most frequent N actions
        num_embedding = sum(df_metric_cat['count'] >= 20) # only words occurring at least 20 times
        # num_embedding = len(df_metric_cat) # num words in annotated vocab (not entire vocab)
    
    inv_token_dict = {v: k for k, v in token_dict.items()}
    df_metric_cat["token"] = df_metric_cat["action"].map(inv_token_dict)
    df_metric_cat = df_metric_cat[~df_metric_cat["token"].isna()] # 3 actions unable to be matched to tokens for some reason...

    if num_embedding < df_metric_cat.shape[0]:
        word_list = df_metric_cat['token'][:num_embedding].tolist()
    else:
        word_list = wv.index2word

    # Fit TSNE
    X = wv[word_list]
    model_tsne = TSNE(n_components=2,
                      init='pca',
                      random_state=123,
                      method='barnes_hut',
                      perplexity= perplexity,
                      n_iter=n_iter,
                      verbose=2)
    Y = model_tsne.fit_transform(X)

    # Map TSNE to actions and categories
    tsne_out = pd.concat([pd.DataFrame(Y, columns=["x1", "x2"]), 
                          pd.Series(word_list, name="token")], axis=1)
    tsne_out = pd.merge(tsne_out, df_metric_cat, how="left", on="token")

    return tsne_out[['token', 'x1', 'x2', 'action', 'metric_category', 'count', 'time_diff']]

def plot_tsne(tsne_out, plot_nan = False, plot_ticks = False, all_labels = False):
    ''' Plot representation of tsne contained in tsne_out.
        plot_nan - whether to plot actions with missing category annotations
        plot_ticks - whether to plot x/y ticks and labels
        all_labels - whether to plot all categories, or only a select subset
    '''
    if all_labels:
        label_list = tsne_out['metric_category'].unique().tolist()
    else:
        label_list = ['Chart Review', 'Note Review', 'Results Review', 'BPA',
                      'Navigation', 'Order Entry', 'Inbox', 'Note Entry']
    colors = cm.rainbow(np.linspace(0, 1, len(label_list)))

    def _plot(label, color, alpha=0.8, linewidth=1):
        plt.scatter(tsne_out[tsne_out.metric_category == label].x1,
                tsne_out[tsne_out.metric_category == label].x2,
                marker='o',
                color=color, #'lightsteelblue',
                linewidth=linewidth, #1
                alpha=alpha, #0.8
                label=label)

    plt.figure(figsize=(7, 7))

    if plot_nan:
        plt.scatter(tsne_out[pd.isna(tsne_out.metric_category)].x1,
                    tsne_out[pd.isna(tsne_out.metric_category)].x2,
                    marker='o', color='black',
                    linewidth=1, alpha=0.3,
                    label='Unknown')

    _plot('Chart Review', 'lightsteelblue')
    for l, c, co in zip(label_list, colors, range(len(label_list))):
        if l == 'Chart Review' or pd.isna(l):
            continue
        _plot(l, c, linewidth=1.5)
    
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=plot_ticks,  # ticks along the bottom edge are off
        top=plot_ticks,  # ticks along the top edge are off
        labelbottom=plot_ticks)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y', which='both', left=plot_ticks, right=plot_ticks,  
        labelleft=plot_ticks)
    plt.legend(loc='best')
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('./result/tsne.pdf')
    plt.show()


def get_neighbors(action_name, wv, token_dict):
    ''' For a particular action_name (type str), print top 10 nearest neighbors
    '''
    inv_token_dict = {v: k for k, v in token_dict.items()}
    token = inv_token_dict[action_name]
    top = wv.most_similar(positive=[str(token)], topn=10)
    top_list = [(x[0], token_dict[x[0]], x[1]) for x in top]
    print('Token:', token, token_dict[str(token)])
    print(top_list, '\n')

def identify_clusters(query_string, tsne_out):
    ''' Identify clusters by their x and y coordinates
        query string must be in the format 'x1 > _ and x2 < _'
    '''
    tsne_out.query(query_string).to_csv('./result/tsne_cluster.csv', index=False)

def compute_topN_proximity(wv, token_dict, topN = 10):
    ''' For each word in vocabulary, compute the mean, min, and max distance of
        its nearest topN neighbors, write to csv file
    '''
    inv_token_dict = {v: k for k, v in token_dict.items()}
    output = []
    for k in inv_token_dict:
        token = inv_token_dict[k]
        top = wv.most_similar(positive=[str(token)], topn=topN)
        top = pd.DataFrame(top, columns=['token', 'distance'])
        best = top.distance.max()
        worst = top.distance.min()
        avg = top.distance.mean()
        output.append((token, k, avg, best, worst))
    output = pd.DataFrame(output, columns=['token', 'name', 'mean', 'max', 'min'])
    output = output.sort_values(by='mean', ascending=False)
    output.to_csv('./result/embedding_distances.csv', index=False)

def find_category_membership(category, df_metric_cat, token_dict):
    ''' Returns a list of tokens that match a category string in metric_category
    '''
    inv_token_dict = {v: k for k, v in token_dict.items()}
    df_metric_cat["token"] = df_metric_cat["action"].map(inv_token_dict)
    return df_metric_cat[df_metric_cat.metric_category == category].token.to_list()

def compute_group_similarity(wv, group_list, raw_wv = True):
    ''' Calculate distances within / outside a group
        group - list of str tokens
        wv - either a gensim wv object, or a dataframe with columns named x1, x2 (from tsne)
    '''
    if raw_wv: # raw gensim object
        word_list = wv.index2word
        dist_matrix = sklearn.metrics.pairwise_distances(wv[word_list])
    else: # pd dataframe
        word_list = [str(i) for i in wv.token.to_list()]
        dist_matrix = sklearn.metrics.pairwise_distances(wv[["x1", "x2"]])

    group_distances = []
    for token in group_list:
        if not pd.isna(token):
            try:
                i = word_list.index(token)
                mean_dist = np.median(dist_matrix[i, :])
                group_distances.append(mean_dist)
            except:
                continue
                # df_metric_cat includes all tokens, but tsne_out only has the common ones
                # print('token', token, 'not found')
    group_distances = pd.Series(group_distances, name='in_group')

    other_distances = []
    other_tokens = set(word_list).difference(set(group_list))
    for token in other_tokens:
        i = word_list.index(token)
        mean_dist = np.median(dist_matrix[i, :])
        other_distances.append(mean_dist)
    other_distances = pd.Series(other_distances, name='out_group')

    return group_distances.mean()/other_distances.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', type=str, default='action_embedding')
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--context', type=int, default=10)
    parser.add_argument('--if_train', type=boolean_string, default=False)
    parser.add_argument('--if_plot', type=boolean_string, default=False)
    parser.add_argument('--if_metric_alone', type=boolean_string, default=False)
    parser.add_argument('--if_neighbors', type=boolean_string, default=False)
    args = parser.parse_args()
    print(args)

    # Paths
    metric_cat = './aux_data/metric_categorized.csv'
    df_metric_cat = pd.read_csv(metric_cat)

    if args.if_metric_alone:
        corpus_file = "./aux_data/corpus_metric_name.pkl"
        dict_file = './aux_data/token_dict_metric_name.pkl'
        word_vector_file = './aux_data/metric_vectors_' + str(args.embedding_size) + '.wv'
        df_metric_cat['action'] = df_metric_cat['METRIC_NAME'].map(str)
        # Aggregate counts across the same METRIC_NAME
        df_metric_cat = df_metric_cat.groupby(by=["action", "metric_category"], dropna=False).sum().reset_index()
    else:
        dict_file = './aux_data/token_dict.pkl'
        corpus_file = "./aux_data/corpus.pkl"
        word_vector_file = './aux_data/action_vectors_' + str(args.embedding_size) + '.wv'
        df_metric_cat['action'] = df_metric_cat['METRIC_NAME'].map(str) + '-' + df_metric_cat['REPORT_NAME'].map(str)

    metric_dict = dict(zip(df_metric_cat['action'].values, df_metric_cat['metric_category']))


    # Load processed corpus (of actions)
    with open(corpus_file, 'rb') as f:
        corpus = pickle.load(f)
    with open(dict_file, 'rb') as f:
        token_dict = pickle.load(f)

    # Train action embeddings
    if args.if_train:
        wv = train_embeddings(corpus, args)
        wv.save(word_vector_file)

    # Load saved embeddings
    wv = KeyedVectors.load(word_vector_file)

    # Plot tSNE visualization
    if args.if_plot:
        tsne_out = compute_tsne(wv, df_metric_cat, metric_dict, token_dict,
                                perplexity=30, n_iter=5000, all_words=False)
        plot_tsne(tsne_out, plot_nan = True, plot_ticks = True, all_labels=False)
        tsne_out.to_csv("./result/tsne_out.csv", index=False)

    # Print nearest neighbors for a given action
    if args.if_neighbors:
        # action_name = 'Chart Review Note report viewed-Lactation Note'
        action_name = 'Inpatient system list accessed-nan'
        get_neighbors(action_name, wv, token_dict) # print nearest 10 neighbors

        compute_topN_proximity(wv, token_dict, topN = 10) # compute average similarity metric for all actions

        print(identify_clusters('x1 > 5 and x2 > 10', tsne_out)) # example to pull actions from a certain location in tSNE plot

