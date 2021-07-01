import os
import pickle
import numpy as np
import pandas as pd
from src.data.clustering_models import *
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE



map_categories = {'care.virtue': 'harm', 'care.vice': 'harm', 'authority.virtue': 'authority',
                  'authority.vice': 'authority', 'loyalty.virtue': 'loyalty', 'loyalty.vice': 'loyalty',
                  'fairness.virtue': 'fairness', 'fairness.vice': 'fairness', 'sanctity.virtue':'purity', 'sanctity.vice':'purity', 'MoralityGeneral': 'general'}




models = ['Kmeans', 'GMM']
categories = ['harm', 'fairness', 'authority', 'purity', 'loyalty']


def select_childes(data: pd.DataFrame, identity : str, age: int, category : str, exclude_hall : bool = True) -> list:

    if exclude_hall:
        data = data.loc[data.corpus != 'Hall']

    categories = []
    for foundation, mapped_foundation in map_categories.items():
        if mapped_foundation == category:
            categories.append(foundation)


    new_data = list(data.loc[data.identity == identity].loc[data.year.astype(int) == age].loc[data.category.isin(categories)].context)
    return new_data

def get_embeddings(data: list) -> np.array:
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    vectors = np.array([model.encode(x) for x in data])
    return vectors




def assign_labels(labels, data, age_compressed_vectors,probs, has_probs = False):
    unique_labels = set(labels)
    labels_utterances = {l: [] for l in unique_labels}

    if has_probs:
        for l, d, p, v in zip(labels, data, probs, age_compressed_vectors):
            labels_utterances[l].append((d, p, v))
    else:

        for l, d, v in zip(labels, data, age_compressed_vectors):
            labels_utterances[l].append((d, [], v))


    return labels_utterances


def prepare_visualization(vectors_of_ages):

    """
    Important Note:
        The t-SNE compression should be done for the category on all the ages, because we want to compare the shifts of topics over time.
         
    
    :param data:
    :param vectors:
    :param labels:
    :param model:
    :return:
    """

    data_lens = [len(d) for d in vectors_of_ages]

    # all_data = np.array(reduce(operator.concat, vectors_of_ages))

    all_data = np.concatenate(vectors_of_ages)


    compressed = TSNE(n_components=2).fit_transform(all_data)
    print(np.shape(vectors_of_ages), np.shape(all_data), np.shape(compressed))
    compressed_vectors_of_ages = []
    for i , l in enumerate(data_lens):

        compressed_vectors_of_ages.append(compressed[:l])

        compressed = compressed[l:]

    return compressed_vectors_of_ages

def train_clustering_model(name, data):
    if name == 'Kmeans':
        model= KmeansModel()

    elif name == 'GMM':
        model = GMM()



    vectors = get_embeddings(data)


    best_k, labels, best_clustering_model = model.find_k(vectors)





    return best_k, labels,model, vectors



def category_cluster_by_age(childes_data: pd.DataFrame, category, model_name):
    ages = range(1, 7)
    age_datats = []
    vectors_of_ages = []
    for age in ages:
        data = select_childes(childes_data, 'child', age, category)

        best_k,labels, model, vectors = train_clustering_model(model_name, data)
        age_datats.append((best_k,labels, model, vectors,data))
        vectors_of_ages.append(vectors)


    compressed_vectors = prepare_visualization(vectors_of_ages)

    for i, age in enumerate(ages):
        best_k, labels, model, vectors,data = age_datats[i]
        age_compressed_vectors = compressed_vectors[i]

        if model_name == 'GMM':
            probs = model.get_probs(vectors)

            labels_utterances = assign_labels(labels, data, age_compressed_vectors,probs,has_probs=True)

        else:
            probs = None
            labels_utterances = assign_labels(labels, data, age_compressed_vectors,probs, has_probs=False)

        pickle.dump((labels_utterances, probs, best_k, model), open(os.path.join('data', model_name, category, f'{age}.pkl'), 'wb'))


def category_cluster(childes_data: pd.DataFrame, category, model_name):
    all_utterances =  []
    age_utterances = []
    age_vectors = []
    ages = range(1, 7)
    for age in ages:
        data = select_childes(childes_data, 'child', age, category)
        vectors= get_embeddings(data)

        age_utterances.append(data)
        age_vectors.append(vectors)

        all_utterances += data

    best_k, labels, model, vectors = train_clustering_model(model_name, all_utterances)
    compressed_vectors = prepare_visualization(age_vectors)

    for i, age in enumerate(ages):
        data = age_utterances[i]
        v = age_vectors[i]
        age_compressed_vectors = compressed_vectors[i]

        age_labels = model.predict(v)


        if model_name == 'GMM':
            probs = model.get_probs(v)

            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=True)

        else:
            probs = None
            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=False)


        pickle.dump((labels_utterances, probs, best_k, model),
                    open(os.path.join('data', model_name, category, f'{age}_all.pkl'), 'wb'))


if __name__ == '__main__':
    df = pickle.load(open('data/pickled-data/moral_df_context_2.p', 'rb'))


    #
    # for m in models:
    #     for c in categories:
    #         category_cluster_by_age(df,c, m)


    for m in models:
        for c in categories:
            category_cluster(df, c, m)








