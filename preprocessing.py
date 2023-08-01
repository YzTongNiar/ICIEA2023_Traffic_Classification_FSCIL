import numpy as np
import pandas as pd
from tqdm import tqdm


def time_difference(time1, time2):
    list1 = list(map(float, time1.split(":")))
    list2 = list(map(float, time2.split(":")))
    time_diff = 60 * (list2[0] - list1[0]) + (list2[1] - list1[1])
    return time_diff


def feature_encoder(feature_sequences):
    encoded_feature_sequence = []
    feature_after_encoding = pd.DataFrame({'time': [],
                                           'mean_tbs_dl': [], 'mean_tbs_ul': [],
                                           'max_tbs_dl': [], 'max_tbs_ul': [],
                                           'total_dl': [], 'total_ul': [],
                                           'num_of_packets_dl': [], 'num_of_packets_ul': []})
    print("Start data encoding")
    for index, feature in enumerate(tqdm(feature_sequences)):
        ifeature_after_encoding = pd.DataFrame({'time': [],
                                                'mean_tbs_dl': [], 'mean_tbs_ul': [],
                                                'max_tbs_dl': [], 'max_tbs_ul': [],
                                                'total_dl': [], 'total_ul': [],
                                                'num_of_packets_dl': [], 'num_of_packets_ul': []})
        ifeature = feature[0]
        itime = ifeature[['time']]
        index1 = ifeature.index.tolist()[0]
        time_slice = 0.1
        time1 = itime.loc[index1]
        for index2, time2 in itime.iterrows():

            if time_difference(time1[0], time2[0]) > 0.1:

                d = {'time': [time_slice],
                     'mean_tbs_dl': [ifeature.loc[index1:index2, ['tbs_dl']].mean()[0]],
                     'mean_tbs_ul': [ifeature.loc[index1:index2, ['tbs_ul']].mean()[0]],
                     'max_tbs_dl': [ifeature.loc[index1:index2, ['tbs_dl']].max()[0]],
                     'max_tbs_ul': [ifeature.loc[index1:index2, ['tbs_ul']].max()[0]],
                     'total_dl': [ifeature.loc[index1:index2, ['tbs_dl']].sum()[0]],
                     'total_ul': [ifeature.loc[index1:index2, ['tbs_ul']].sum()[0]],
                     'num_of_packets_dl': [
                         (ifeature.loc[index1:index2, ['tbs_dl']] != 0).astype(int).sum(axis=0)[0]],
                     'num_of_packets_ul': [
                         (ifeature.loc[index1:index2, ['tbs_ul']] != 0).astype(int).sum(axis=0)[0]]}
                d1 = pd.DataFrame(d)
                # d1 = d1.replace([np.inf, -np.inf], 0)
                d1 = d1.fillna(0)
                ifeature_after_encoding = pd.concat([ifeature_after_encoding, d1])
                time_slice = time_slice + round(time_difference(time1[0], time2[0]), 1)
                index1 = index2
                time1 = itime.loc[index1]
        encoded_feature_sequence.append((ifeature_after_encoding, feature[1]))
        ifeature_after_encoding['series_id'] = np.ones(ifeature_after_encoding.shape[0]) * index
        feature_after_encoding = pd.concat([feature_after_encoding, ifeature_after_encoding])
    return encoded_feature_sequence, feature_after_encoding
