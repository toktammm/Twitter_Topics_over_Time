import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import pandas as pd
import csv
from pandarallel import pandarallel
pandarallel.initialize()
import copy
import pickle
from pandas.core.common import flatten
import numpy as np
from itertools import combinations

def clean(text):
    text = str(text)
    return text.replace('[','').replace(']','').replace('"','').replace("'",'').replace(',',' ').split()

def main():
    coh_k = 500      
    alpha = 1
    beta = 0.5

    pickle_path = input('input path to the noc pickle file:')
    tw_data = input('input path to the twitter input file:')
    df = pd.read_csv(tw_data)

    with open(pickle_path, 'rb') as f:
        par = pickle.load(f)
    phi = par['n']
    n_t = par['T']

    word_topic_ind = []
    word_topic_phi = []
    topic_words = []
    topic_important_words = []
    topic_important_words_probs = []
    topic_phi_sort_ind = []
    for i in range(n_t):
        word_topic_ind.append(np.array(np.nonzero(phi[i]))[0])
        word_topic_phi.append(phi[i][np.nonzero(phi[i])])

    for i in range(n_t):
        temp = []
        for j in word_topic_ind[i]:
            temp.append(par['word_token'][j])
        topic_words.append(temp)
    word_probs = []
    for i in range(n_t):
        topic_phi_sort_ind.append(np.argsort(word_topic_phi[i])[:-coh_k-1:-1])
        wordtemp = []
        probtemp = []
        for j in topic_phi_sort_ind[i]:  # find 500 most important words (ids) of each topic
            wordtemp.append(topic_words[i][j])
            probtemp.append(word_topic_phi[i][j])
        
        topic_important_words.append(wordtemp)
        topic_important_words_probs.append(probtemp)

    text = df.text.apply(clean).values

    print('calculating the coherence ')
    all_words = list(set(flatten(topic_important_words)))
    d_dict = {i: [] for i in all_words}
    for d in range(len(all_words)):
        ind_count = []
        for i in range(len(df)):
            if all_words[d] in my_data[i]:
                d_dict[all_words[d]].append(i)
                ind_count.append(i)

    comb = list(combinations(list(range(coh_k)), 2))
    coh = []
    for t in range(n_t):
        t_coh = []
        for c in range(len(comb)):
            w1 = topic_important_words[t][comb[c][0]]
            p1 = topic_important_words_probs[t][comb[c][0]]
            w2 = topic_important_words[t][comb[c][1]]
            p2 = topic_important_words_probs[t][comb[c][1]]
            intersection = [value for value in d_dict[w1] if value in d_dict[w2]]
            joint = len(intersection) / len(df)
            if not joint:
                joint = 1e-10
            p_prod = p1 * p2
            t_coh.append(np.log2(joint / p_prod))
        coh.append(np.mean(t_coh))

    coherence = np.mean(coh)
    print('topics: ', str(n_t), 'alpha: ', alpha, 'beta: ', beta, 'Coherence Score: ', coherence)


if __name__ == "__main__":
    main()
