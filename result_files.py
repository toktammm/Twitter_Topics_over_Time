import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axis, lines
import scipy.stats
from scipy.stats import beta
import pprint, pickle
import matplotlib.cm as cm
import csv
import os
import copy
import sys
from pandas.core.common import flatten
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from pandarallel import pandarallel
pandarallel.initialize()
csv.field_size_limit(sys.maxsize)


def clean(text):
    return str(text).replace('[', '').replace(']', '').replace("'", "").split()

def clean_id(text):
    return str(text).replace('[', '').replace(']', '').replace("'", "").split(',')

def clean_users(text):
    if type(text)==list:
        text = str(list(flatten(text)))
    return text.replace('[','').replace(']','').replace(' ','').replace('"','').replace("'",'').replace(',',' ').split()

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def VisualizeTopics(resultspath, phi, words, num_topics, n_iteration, viz_threshold=5e-3):
    phi_viz = np.transpose(phi)
    words_to_display = ~np.all(phi_viz <= viz_threshold, axis=1)
    phi_viz = phi_viz[words_to_display]

    fig, ax = plt.subplots()
    heatmap = plt.pcolor(phi_viz, cmap=cm.Blues, alpha=0.8)
    plt.colorbar()

    ax.grid(False)
    ax.set_frame_on(False)

    ax.set_xticks(np.arange(phi_viz.shape[1]) + 0.5, minor=False)   # from 0 to number of topics
    ax.set_yticks(np.arange(phi_viz.shape[0]) + 0.5, minor=False)   # from 0 to number of words
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    column_labels = [words[i] for i in range(len(words_to_display)) if words_to_display[i]]
    row_labels = ['Topic ' + str(i) for i in range(1, num_topics + 1)]
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig(resultspath + str(num_topics) + 'topics' + str(n_iteration) + 'iteration_' + 'word_distribution_topics.png')
    plt.close()

# get words per topic and phi values per words
def TopicWords(resultspath, phi, num_topics, n_iteration, words):
    word_topic_ind = []
    word_topic_phi = []
    topic_words = []
    topic_important_words = []
    topic_important_words_probs = []
    topic_phi_sort_ind = []
    for i in range(num_topics):
        word_topic_ind.append(np.array(np.nonzero(phi[i]))[0])
        word_topic_phi.append(phi[i][np.nonzero(phi[i])])

    for i in range(num_topics):
        temp = []
        for j in word_topic_ind[i]:
            temp.append(words[j])
        topic_words.append(temp)
    with open(resultspath + str(num_topics) + 'topics' + str(n_iteration) + 'iteration_' +'all_topic_words.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(topic_words)
    word_probs = []
    for i in range(num_topics):
        # sorting most important words per topic (with higher phi values)
        topic_phi_sort_ind.append(np.argsort(word_topic_phi[i])[:-100:-1])
        wordtemp = []
        probtemp = []
        for j in topic_phi_sort_ind[i]:  # find 100 most important words (ids) of each topic
            wordtemp.append(topic_words[i][j])
            probtemp.append(word_topic_phi[i][j])
        
        topic_important_words.append(wordtemp)
        topic_important_words_probs.append(probtemp)
        # dictionary of topic words and their probabilities 
        word_probs.append([dict(zip(wordtemp, probtemp))])
    
    with open(resultspath + str(num_topics) + 'topics' + str(n_iteration) + 'iteration_' + 'word_probs.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(word_probs)   
    with open(resultspath + str(num_topics) + 'topics' + str(n_iteration) + 'iteration_' + 'topic_important_words_sep.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(topic_important_words)
    with open(resultspath + str(num_topics) + 'topics' + str(n_iteration) + 'iteration_' + 'topic_important_words_probs_sep.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(topic_important_words_probs)


def VisualizeEvolution(result_path, result_folder, par, num_topics, n_iteration, n_partitions, data_path, result_file):
    psi = copy.deepcopy(par['psi'])
    times = pd.read_csv(result_path+result_file+'t_dict.csv', names=['timestamp', 'date'])
    times_timestamps = times.timestamp.values
    times_dates = times.date.values

    print('reading input files')
    df = pd.read_csv(data_path)
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)

    t_ids = np.array(df.userid.apply(clean_id).apply(len))  # number of aggregated tweets per row in df
    DayGroupedDf = pd.DataFrame(df.groupby(['date']).userid.apply(list))
    DayGroupedDf = DayGroupedDf.reset_index(drop=False)
    DayGroupedDf.userid = DayGroupedDf.userid.apply(clean_users)
    y = np.array(DayGroupedDf.userid.str.len())
    x = np.array(DayGroupedDf.date)
    plt.style.use('seaborn-white')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    plt.plot(x, y, linestyle='solid')
    plt.savefig(result_path + result_folder + 'activity.png')
    plt.cla()

    new_timestamps = [par['t'][i][0] for i in range(len(par['t']))]
    t_part = copy.deepcopy(par['t_part'])

    n_active_ids_partition = []
    for ind in range(len(t_part)):
        if ind == len(t_part) - 1:
            temp_inx = np.where(np.array(new_timestamps) >= t_part[ind])
            new_timestamps = np.array(new_timestamps)
        else:
            temp_inx = np.where(
                (np.array(new_timestamps) >= t_part[ind]) & (np.array(new_timestamps) <= t_part[ind + 1]))
            new_timestamps = np.array(new_timestamps)
        temp_ids = t_ids[temp_inx]  
        n_active_ids_partition.append(temp_ids.sum())

    b_timestamp = []
    for i in range(len(t_part)):
        ind = np.where(times_timestamps == t_part[i])
        if len(ind[0]) == 0:
            ind = find_nearest(times_timestamps, t_part[i])
            b_timestamp.append(times_dates[ind])
        else:
            b_timestamp.append(times_dates[ind][0])
    b_timestamp.append(times_dates[-1])

    n_active_ids_partition = np.insert(n_active_ids_partition, 0, 0)
    n_active_ids_partition = np.append(n_active_ids_partition, 0)
    n_active_ids_partition = np.array(n_active_ids_partition) / sum(n_active_ids_partition)
    for i in range(0, num_topics):
        if i == 0:
            plt.ylim(0, 0.3)
            plt.step(range(len(n_active_ids_partition)), n_active_ids_partition / sum(n_active_ids_partition),
                     linewidth=2, color='red')  
            plt.xticks(np.array(range(len(psi[i]))), b_timestamp,
                       rotation='vertical') 
            plt.tick_params(axis='x', which='major', labelsize=7)
            plt.grid(axis='x', color='0.95')
            plt.grid(axis='y', color='0.95')
            plt.savefig(result_path + result_folder + str(num_topics) + 'topic' + '_' + str(
                n_iteration) + 'iteration_' + 'activity.png')
            plt.ylim(0, 0.3)
            plt.step(range(len(n_active_ids_partition)), n_active_ids_partition / sum(n_active_ids_partition),
                     linewidth=2, color='red')  
            plt.xticks(np.array(range(len(psi[i]))), b_timestamp,
                       rotation='vertical') 
            plt.tick_params(axis='x', which='major', labelsize=7)
            plt.grid(axis='x', color='0.95')
            plt.grid(axis='y', color='0.95')
            plt.savefig(result_path + result_folder + str(num_topics) + 'topic' + '_' + str(
                n_iteration) + 'iteration_' + 'activity.png')
            plt.close()
            plt.close()
        plt.ylim(0, 1.1)
        psi[i] = np.insert(psi[i], 0, 0)
        psi[i] = np.append(psi[i], 0)
        plt.step(range(len(psi[i])), np.array(psi[i]).transpose(), linewidth=2)
        plt.xticks(np.array(range(len(psi[i]))), b_timestamp, rotation='vertical')
        plt.tick_params(axis='x', which='major', labelsize=7)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.savefig(result_path + result_folder + str(num_topics) + 'topic' + str(i) + '_' +
                    str(n_iteration) + 'iteration_' + 'topic_dictributions_time.png')
        plt.close()

    for i in range(0, num_topics):
        plt.ylim(0, 1.1)
        plt.step(range(len(psi[i])), np.array(psi[i]).transpose(), linewidth=2)
        plt.xticks(np.array(range(len(psi[i]))), b_timestamp, rotation='vertical')
        plt.tick_params(axis='x', which='major', labelsize=7)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
    plt.savefig(result_path + result_folder + str(num_topics) + 'topic' + '_' +
                str(n_iteration) + 'iteration_' + 'all_topic_dictributions_time.png')
    plt.close()


def evaluation(randomint, num_topics, n_iter, n_partition, result_path, result_folder, data_path, par):

    tempdf = pd.read_csv(data_path)
    mydictionary = copy.deepcopy(par['word_token'])

    result_file = result_path + result_folder + str(num_topics) + 'topics' + str(
        n_iter) + 'iteration_' + 'topic_important_words_sep.csv'  
    wordtemp = []
    with open(result_file) as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            wordtemp.append(row)

    rsult_file = result_path + result_folder + '/' + str(num_topics) + 'topics' + str(
        n_iter) + 'iteration_' + 'topic_important_words_probs_sep.csv'  
    probtemp = []
    with open(result_file) as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            probtemp.append(row)

    all_dic = []
    for i in range(len(wordtemp)):  # for number of topics
        wordtemp[i] = str(wordtemp[i]).replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(',',' ').split()
        probtemp[i] = str(probtemp[i]).replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(',',' ').split()
        probtemp[i] = np.array(probtemp[i]).astype(float)
        all_dic.append(dict(zip(wordtemp[i], probtemp[i])))

    id2word = corpora.Dictionary([list(mydictionary)])
    my_data = tempdf.nofreq.apply(lambda x: x.replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(' ', '').replace(',', ' ').split()).values
    my_topics_ls = [list(all_dic[i].keys()) for i in range(len(all_dic))]

    cm = CoherenceModel(topics=my_topics_ls, texts=my_data, dictionary=id2word, coherence='c_v') 
    coherence = cm.get_coherence()
    print('Coherence Score: ', coherence)

def main():
    alpha = 1
    beta = 0.01
    n_iteration = 100
    n_partitions = 'biweek'     # month/biweek
    randomint = True            # True/False

    data_path = input('enter path to data file:')
    result_path  = input('enter path to save results:'+'/')

    # results path to save files (visualization of results)
    result_folder = str(par['T']) + 'topics' + str(n_iteration) + 'iterations' + n_partitions + '/'

    if  randomint:    # random initialization using org_text data
        result_path = result_path + 'tw_orgtext/' + 'randomint/'+ str(alpha)+'_'+str(beta)+'_'+str(par['T'])+'_'+str(n_iteration)+'/'
        result_file = 'orgtext_randomint_'
    else:             # activity-based initialization using org_text data
        result_path = result_path + 'tw_orgtext/' + 'activityint/'+ str(alpha)+'_'+str(beta)+'_'+str(par['T'])+'_'+str(n_iteration)+'/'
        result_file = 'orgtext_activity_'

    os.makedirs(result_path + result_folder, exist_ok=True)
    tot_pickle_path = result_path + str(par['T']) + 'topics' + str(n_iteration) + 'iteration_' + 'result_tot.pickle'
    tot_pickle = open(tot_pickle_path, 'rb')
    par = pickle.load(tot_pickle)
    
    VisualizeTopics(result_path + result_folder, par['n'], par['word_token'], par['T'], n_iteration)
    TopicWords(result_path + result_folder, par['n'], par['T'], n_iteration, par['word_token'])
    VisualizeEvolution(result_path , result_folder, par, par['T'], n_iteration, n_partitions, data_path, result_file)
    evaluation(randomint, par['T'], n_iteration, n_partitions, result_path, result_folder, data_path, par)


if __name__ == "__main__":
    main()
