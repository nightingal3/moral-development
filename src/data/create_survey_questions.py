import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import operator
from scipy.spatial import distance

import warnings

categories = ['harm', 'authority', 'fairness', 'loyalty', 'purity']
ages = range(1, 7)


def get_df(labels_utterances, age, has_probs=False):
    df = pd.DataFrame()
    columns = ['label', 'context', 'x', 'y']
    labels = []
    context = []
    x = []
    y = []
    for l, d in labels_utterances.items():
        for c, probs, pos in d:

            labels.append(str(l))
            context.append(c)
            x.append(pos[0])
            y.append(pos[1])
            if has_probs:
                # TODO
                pass

    df['label'] = labels
    df['context'] = context
    df['x'] = x
    df['y'] = y
    df['age'] = [age] * len(df)
    return df


def find_distance_to_center(center_x, center_y):
    def function(row):
        x = row['x']
        y = row['y']
        center = np.array([center_x, center_y])
        pos = np.array([x, y])

        cos_distance = distance.cosine(center, pos)

        return cos_distance

    return function


def get_cluster_documents_from_label(df, label, identity, n=10, l=15, m=10):
    '''
    df is a dataframe sorted based on distance to center
    '''
    closest_context = list(set(df.iloc[:n]['context']))

    farthest = list(set(df.iloc[-l:]['context']))
    farthest = [x for x in farthest if x not in closest_context]

    closest_context = [f'{identity} says: {x}' for x in closest_context]
    farthest = [f'{identity} says: {x}' for x in farthest]

    random_choices = np.random.choice(len(farthest), min(len(farthest), m), False)

    far_context = []
    for random_idx in random_choices:
        context = farthest[random_idx]
        far_context.append(context)

    new_df = pd.DataFrame()
    new_df['utterance type'] = ['central'] * len(closest_context) + ['peripheral'] * len(far_context)
    new_df['utterance'] = closest_context + far_context
    new_df['label'] = [label] * len(new_df)
    return new_df


def get_cluster_documents(df, identity, n=10, l=15, m=10):
    labels = df.label.unique()
    all_df = pd.DataFrame()
    for l in labels:
        new_df = df.loc[df.label == l]
        center_x = np.array(new_df['x']).mean()
        center_y = np.array(new_df['y']).mean()

        distance_function = find_distance_to_center(center_x, center_y)
        new_df['distance'] = new_df.apply(distance_function, axis=1)
        new_df = new_df.sort_values(by='distance')


        all_df = all_df.append(get_cluster_documents_from_label(new_df, identity=identity, label=l, ))


    return all_df


def find_dimension(identity, category, sentiment):
    file_name = f'../data/dimensionality_reduction/childes_{identity}_{category}_{sentiment}'
    df = pd.read_csv(f"{file_name}.csv")

    for i in range(len(df)):
        if i > 0 and float(df.loc[i - 1]) < 0.95 and float(df.loc[i]) >= 0.95:
            return i + 1


all_clusters_df = pd.DataFrame()
model_name = 'GMM'
identity = 'child'
for c in categories:
    for p in ['pos', 'neg']:
        all_df = pd.DataFrame()
        for age in ages:
            moral_dir = f'../data/{model_name}/{identity}/{c}-{p}/{age}_all_dimension_reduced.pkl'
            if not os.path.exists(moral_dir):
                continue
            dimension = find_dimension(identity, c, p)
            labels_utterances, probs, best_k, model = pickle.load(
                open(f'../data/{model_name}/{identity}/{c}-{p}/{age}_all_dimension_reduced.pkl', 'rb'))
            df = get_df(labels_utterances, age)

            all_df = all_df.append(df)
        print(best_k)

        all_df = get_cluster_documents(all_df, identity)
        all_df['foundation'] = [c + ' ' + p] * len(all_df)
        all_df['best k'] = [best_k] * len(all_df)
        all_df['identity'] = [identity] * len(all_df)
        all_df['model'] = [model_name] * len(all_df)
        all_df['dimension'] = [dimension] * len(all_df)

        all_clusters_df = all_clusters_df.append(all_df)

identity = 'parent'

for c in categories:
    for p in ['pos', 'neg']:
        all_df = pd.DataFrame()
        for age in ages:
            moral_dir = f'../data/{model_name}/{identity}/{c}-{p}/{age}_all_dimension_reduced.pkl'
            if not os.path.exists(moral_dir):
                continue
            dimension = find_dimension(identity, c, p)
            labels_utterances, probs, best_k, model = pickle.load(open(f'../data/{model_name}/{identity}/{c}-{p}/{age}_all_dimension_reduced.pkl', 'rb'))
            df = get_df(labels_utterances, age)

            all_df = all_df.append(df)
        print(best_k)


        all_df = get_cluster_documents(all_df, 'caretaker')
        all_df['foundation'] = [c + ' '+ p] * len(all_df)
        all_df['best k'] = [best_k] * len(all_df)
        all_df['identity'] = [identity] * len(all_df)
        all_df['model'] = [model_name] * len(all_df)
        all_df['dimension'] = [dimension] * len(all_df)

        all_clusters_df = all_clusters_df.append(all_df)

all_clusters_df = all_clusters_df.reset_index()
all_clusters_df['sentence_id']  = all_clusters_df.index
sample_df = all_clusters_df[['utterance', 'identity', 'sentence_id']]
child_samples = sample_df.loc[sample_df.identity == 'child']
parent_samples = sample_df.loc[sample_df.identity == 'parent']
child_samples.to_csv('../data/child_samples.csv', index = False)
parent_samples.to_csv('../data/parent_samples.csv', index = False)

