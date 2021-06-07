import numpy as np
import pickle
import time
import math
import pickle
import bisect
import copy
import random
import sys
import os
import itertools
import csv
import scipy.special
import pandas as pd
from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from pandas.core.common import flatten
from pandarallel import pandarallel
pandarallel.initialize()

iter_kl_div: list = []

def clean(text):
   return str(text).replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(' ','').replace(',', ' ').split()

def clean_id(text):
   return str(text).replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(' ', '').replace(',', ' ').split()

def kl_divergence(p, q):
   return np.sum(np.where(np.array(p) != 0, p * np.log(np.array(p) / np.array(q)), 0))

def month_ext(timestamp):
   return timestamp.month

def day_ext(timestamp):
   if timestamp.day < 14:
       return 1
   else:
       return 2

def to_timestamp(x):
   return x.total_seconds()


class NOC:
    def GetDataCorpusAndDictionary(self, df):
        # sort timestamps, get index and sort -> entity, non_entity accordingly
        df = df.sort_values(by=['date'])
        df = df.reset_index(drop=True)
        text = df.nofreq.parallel_apply(clean).values
        timestamps = pd.to_datetime(df.date)
        timestamps = timestamps.dt.date.values
        '''change dates to timestamps'''
        time_avg = min(timestamps) + ((max(timestamps) - min(timestamps)) / 2)
        new_timestamps = timestamps - time_avg
        new_timestamps = list(map(to_timestamp, new_timestamps))

        documents = copy.deepcopy(text)
        dictionary = list(set(flatten(text)))

        '''for timestamp in timestamp_file:'''
        first_timestamp = new_timestamps[0]
        last_timestamp = new_timestamps[len(new_timestamps) - 1]
        new_timestamps = [1.0 * (t - first_timestamp) / (last_timestamp - first_timestamp) for t in new_timestamps]
        '''make a dictionary of new and original timestamps'''
        t_dict = {k: v for k, v in zip(new_timestamps, timestamps)}

        assert len(documents) == len(timestamps)
        return documents, new_timestamps, t_dict, dictionary

    def CalculateCounts(self, par):
        for d in range(par['D']):
            for i in range(par['N'][d]):
                topic_di = par['z'][d][i]  # topic in doc d at position i
                word_di = par['w'][d][i]  # word ID in doc d at position i
                par['m'][d][topic_di] += 1
                par['n'][topic_di][word_di] += 1
                par['n_sum'][topic_di] += 1

    def InitializeParameters(self, documents, timestamps, dictionary, iterations, n_t, dataset, alpha, beta):
        par = {}  # dictionary of all parameters
        par['dataset'] = dataset  # dataset name, tw
        par['max_iterations'] = iterations  # max number of iterations in gibbs sampling
        par['T'] = n_t  # number of topics
        par['D'] = len(documents)
        par['V'] = len(dictionary)
        par['N'] = [len(doc) for doc in documents]
        par['alpha'] = [alpha for _ in range(par['T'])]
        par['beta'] = [beta for _ in range(par['V'])]
        par['beta_sum'] = sum(par['beta'])

        par['psi'] = [[1 for _ in range(2)] for _ in range(par['T'])]
        par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]
        par['word_id'] = {dictionary[i]: i for i in range(len(dictionary))}
        par['word_token'] = dictionary
        par['z'] = [[random.randrange(0, par['T']) for _ in range(par['N'][d])] for d in range(par['D'])]
        par['t'] = [[timestamps[d] for _ in range(par['N'][d])] for d in range(par['D'])]
        par['w'] = [[par['word_id'][documents[d][i]] for i in range(par['N'][d])] for d in range(par['D'])]
        par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
        par['n'] = [[0 for v in range(par['V'])] for t in range(par['T'])]
        par['n_sum'] = [0 for t in range(par['T'])]
        np.set_printoptions(threshold=np.inf)
        np.seterr(divide='ignore', invalid='ignore')
        self.CalculateCounts(par)
        return par

    def GetTopicTimestamps(self, par):
        topic_timestamps = []
        for topic in range(par['T']):
            current_topic_timestamps = []
            current_topic_doc_timestamps = [[(par['z'][d][i] == topic) * par['t'][d][i] for i in range(par['N'][d])] for
                                            d in range(par['D'])]
            for d in range(par['D']):
                current_topic_doc_timestamps[d] = filter(lambda x: x != 0, current_topic_doc_timestamps[d])
            for timestamps in current_topic_doc_timestamps:
                current_topic_timestamps.extend(timestamps)
            # assert current_topic_timestamps != []
            if (current_topic_timestamps == []):
                print('topic timestamps vector all zero for topic ', topic)
                print(par['z'])
                sys.exit()
            topic_timestamps.append(current_topic_timestamps)
        return topic_timestamps

    def GetMethodOfMomentsEstimatesForPsi(self, par):
        topic_timestamps = self.GetTopicTimestamps(par)
        psi = [[1 for _ in range(2)] for _ in range(len(topic_timestamps))]
        for i in range(len(topic_timestamps)):
            current_topic_timestamps = topic_timestamps[i]
            timestamp_mean = np.mean(current_topic_timestamps)
            timestamp_var = np.var(current_topic_timestamps)
            if timestamp_var == 0:
                timestamp_var = 1e-4
            common_factor = timestamp_mean * (1 - timestamp_mean) / timestamp_var - 1
            psi[i][0] = 1 + timestamp_mean * common_factor   
            psi[i][1] = 1 + (1 - timestamp_mean) * common_factor   

        return psi

    def ComputePosteriorEstimatesOfThetaAndPhi(self, par):
        theta = copy.deepcopy(par['m'])
        phi = copy.deepcopy(par['n'])

        for d in range(par['D']):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray([1.0 / len(theta[d]) for _ in range(len(theta[d]))])
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0 * theta[d] / sum(theta[d])
        theta = np.asarray(theta)

        for t in range(par['T']):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0 / len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0 * phi[t] / sum(phi[t])
        phi = np.asarray(phi)

        return theta, phi

    def GibbsSampling(self, par):
        for iteration in range(par['max_iterations']):
            for d in range(par['D']):
                for i in range(par['N'][d]):
                    word_di = par['w'][d][i]
                    t_di = par['t'][d][i]

                    old_topic = par['z'][d][i]
                    par['m'][d][old_topic] -= 1
                    par['n'][old_topic][word_di] -= 1
                    par['n_sum'][old_topic] -= 1

                    topic_probabilities = []
                    for topic_di in range(par['T']):
                        if (par['betafunc_psi'][topic_di] == 0) or ((par['n_sum'][topic_di] + par['beta_sum']) == 0):
                            if par['betafunc_psi'][topic_di] == 0:
                                par['betafunc_psi'][topic_di] = 1e-6
                        psi_di = par['psi'][topic_di]
                        topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
                        topic_probability *= ((1 - t_di) ** (psi_di[0] - 1)) * ((t_di) ** (psi_di[1] - 1))
                        topic_probability /= par['betafunc_psi'][topic_di]
                        topic_probability *= (par['n'][topic_di][word_di] + par['beta'][word_di])
                        topic_probability /= (par['n_sum'][topic_di] + par['beta_sum'])
                        topic_probabilities.append(topic_probability)
                    sum_topic_probabilities = sum(topic_probabilities)
                    topic_probabilities = [p / sum_topic_probabilities for p in topic_probabilities]
                    new_topic = list(np.random.multinomial(1, np.asarray(topic_probabilities).astype('float64'),
                                                               size=1)[0]).index(1)
                    par['z'][d][i] = new_topic
                    par['m'][d][new_topic] += 1
                    par['n'][new_topic][word_di] += 1
                    par['n_sum'][new_topic] += 1

                if d % 1000 == 0:
                    print('Done with iteration {iteration} and document {document}'.format(iteration=iteration,
                                                                                           document=d))
            '''upatate par['psi'] with: count in each bean for t for each z'''
            print('update psi for iteration ', iteration)
            par['psi'] = self.GetMethodOfMomentsEstimatesForPsi(par)
            par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]

        print('compute posterior estimates of theta and phi')
        par['m'], par['n'] = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
        return par['m'], par['n'], par['psi']


def main():
    n_iteration = 100
    dataset = 'tw'     
    n_t = 5  # number of topics
    alpha = 1  # 0.1, 0.3, 1
    beta = 0.5

    resultspath = input('where to save results:')  
    resultspath = resultspath + str(alpha) + '_' + str(beta) + '_' + str(n_t) + '_' + str(n_iteration) + '/'
    os.makedirs(resultspath, exist_ok=True)

    data_path = input('path to csv data file:')
    df = pd.read_csv(data_path)

    noc_topic_vectors_path = resultspath + str(n_t) + 'topics' + str(
        n_iteration) + 'iteration_' + 'result_noc_topic_vectors.csv'
    noc_topic_mixtures_path = resultspath + str(n_t) + 'topics' + str(
        n_iteration) + 'iteration_' + 'result_noc_topic_mixtures.csv'
    noc_topic_shapes_path = resultspath + str(n_t) + 'topics' + str(
        n_iteration) + 'iteration_' + 'result_noc_topic_shapes.csv'
    noc_pickle_path = resultspath + str(n_t) + 'topics' + str(n_iteration) + 'iteration_' + 'result_noc.pickle'

    noc = NOC()
    documents, timestamps, t_dict, dictionary = noc.GetDataCorpusAndDictionary(df)
    par = noc.InitializeParameters(documents, timestamps, dictionary, n_iteration, n_t, dataset, alpha, beta)
    theta, phi, psi = noc.GibbsSampling(par)

    print('mcmc finished')
    np.savetxt(noc_topic_vectors_path, phi, delimiter=',')
    np.savetxt(noc_topic_mixtures_path, theta, delimiter=',')
    np.savetxt(noc_topic_shapes_path, psi, delimiter=',')
    with open(resultspath + 'result_dict.csv', 'w') as f:
        for k, v in t_dict.items():
            f.write('{},{}\n'.format(k, v))
    print('saving results to pickle')
    noc_pickle = open(noc_pickle_path, 'wb')
    pickle.dump(par, noc_pickle)
    noc_pickle.close()

if __name__ == "__main__":
    main()