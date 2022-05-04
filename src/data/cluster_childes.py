import os
import pickle
import numpy as np
import pandas as pd
from src.data.clustering_models import *

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


map_categories = {'care.virtue': 'harm', 'care.vice': 'harm', 'authority.virtue': 'authority',
                  'authority.vice': 'authority', 'loyalty.virtue': 'loyalty', 'loyalty.vice': 'loyalty',
                  'fairness.virtue': 'fairness', 'fairness.vice': 'fairness', 'sanctity.virtue': 'purity',
                  'sanctity.vice': 'purity', 'MoralityGeneral': 'general'}

models = ['Kmeans', 'GMM']
categories = ['harm', 'fairness', 'authority', 'purity', 'loyalty']
embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')


def select_childes_with_terms(data: pd.DataFrame, identity: str, age: int, terms: list,
                   exclude_hall: bool = True, reduce = False) -> list:

    if exclude_hall:
        data = data.loc[data.corpus != 'Hall']

    data = data.loc[data.keywords.isin(terms)]



    new_data = data.loc[data.identity == identity].loc[data.year.astype(int) == age]

    if reduce:
        new_data = list(new_data.reduced_context)

    else:
        new_data = list(new_data.context)

    return new_data


def select_childes(data: pd.DataFrame, identity: str, age: int, category: str, sentiment='', use_sentiment=False,
                   exclude_hall: bool = True) -> list:

    if exclude_hall:
        data = data.loc[data.corpus != 'Hall']

    categories = []

    for foundation, mapped_foundation in map_categories.items():

        if mapped_foundation == category:
            categories.append(foundation)

    if use_sentiment:
        new_data = data.loc[data.identity == identity].loc[data.year.astype(int) == age].loc[
                            data.category.isin(categories)].loc[data.sentiment == sentiment]


    else:
        new_data = data.loc[data.identity == identity].loc[data.year.astype(int) == age].loc[
                            data.category.isin(categories)]


    new_data = list(new_data.context)


    return new_data


def select_story_data(data: pd.DataFrame, lower_quantile_val: float, upper_quantile_val: float, category: str) -> list:
    # categories = []
    # for foundation, mapped_foundation in map_categories.items():
    #     if mapped_foundation == category:
    #         categories.append(foundation)
    new_data = list(data.loc[(data.category == category) & ((data["flesch_kincaid"] >= lower_quantile_val) & (
                data["flesch_kincaid"] < upper_quantile_val))].sentence)
    return new_data


def get_embeddings(data: list) -> np.array:
    vectors = embedding_model.encode(data)
    return vectors


def assign_labels(labels, data, age_compressed_vectors, probs, has_probs=False):
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
    for i, l in enumerate(data_lens):
        compressed_vectors_of_ages.append(compressed[:l])

        compressed = compressed[l:]

    return compressed_vectors_of_ages




def train_clustering_model(name, vectors, file_name, constant_dim = False,dim_reduction: int = 200):
    if name == 'Kmeans':
        model = KmeansModel()

    elif name == 'GMM':
        model = GMM()

    # vectors = get_embeddings(data)
    if constant_dim:
        dim  = min(dim_reduction, len(vectors))
        pca = PCA(n_components=dim)
        vectors = pca.fit_transform(vectors)
    else:
        best_dimension = find_best_dimension(file_name)
        dim = best_dimension
        pca = PCA(n_components=best_dimension)
        vectors = pca.fit_transform(vectors)

    best_k, labels, best_clustering_model = model.find_k(vectors)

    return best_k, labels, model, vectors, pca, dim


def find_explained_variance(vectors, out_filename, max_dimensions=300) -> None:
    pca = PCA(n_components=min(max_dimensions, len(vectors)))
    vectors = pca.fit_transform(vectors)
    exp_car_cumul = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, exp_car_cumul.shape[0] + 1), exp_car_cumul)
    plt.xlabel("Number of features")
    plt.ylabel("explained variance")
    plt.savefig(f"{out_filename}.png")
    np.savetxt(f"{out_filename}.csv", exp_car_cumul)


def find_best_dimension(out_file_name):
    df = pd.read_csv(f"{out_file_name}.csv")

    for i in range(len(df)):
        if i > 0 and float(df.loc[i - 1]) < 0.95 and float(df.loc[i]) >= 0.95:
            print(out_file_name, i + 1)
            return i + 1



def category_cluster_generic(childes_data: pd.DataFrame, terms, model_name, identity='child',
                            reduce = True, constant_dim = False, dim = 200):
    all_utterances = []
    age_utterances = []
    age_vectors = []
    ages_includes = []

    ages = range(1, 7)
    first_age = True

    for age in ages:
        data = select_childes_with_terms(childes_data, identity, age, terms,
                              reduce=reduce)

        if len(data) < 1:
            continue
        ages_includes.append(age)
        vectors = get_embeddings(data)

        age_utterances.append(data)
        age_vectors.append(vectors)

        all_utterances += data

        if age == 1 or first_age:
            compiled_vectors = vectors
            first_age  = False
        else:
            compiled_vectors = np.vstack((compiled_vectors, vectors))




    file_name = f'data/dimensionality_reduction/childes_{identity}_{terms}'
    best_k, labels, model, vectors, pca_model, dim = train_clustering_model(model_name, compiled_vectors,file_name, constant_dim, dim_reduction=dim)
    print(best_k)
    compressed_vectors = prepare_visualization(age_vectors)

    for i, age in enumerate(ages_includes):
        data = age_utterances[i]
        v = age_vectors[i]
        age_compressed_vectors = compressed_vectors[i]

        age_labels = model.predict(pca_model.transform(v))

        if model_name == 'GMM':
            probs = model.get_probs(pca_model.transform(v))

            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=True)

        else:
            probs = None
            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=False)



        pickle.dump((labels_utterances, probs, best_k, model),
                    open(os.path.join('data', model_name, identity,'generic', f'{age}_all_dimension_reduced_{terms}.pkl'), 'wb'))





def category_cluster(childes_data: pd.DataFrame, category, model_name, identity='child', sentiment='',
                            use_sentiment=False, constant_dim = False, dim = 200):
    all_utterances = []
    age_utterances = []
    age_vectors = []
    ages_includes = []

    ages = range(1, 7)
    first_age = True

    for age in ages:
        data = select_childes(childes_data, identity, age, category, sentiment=sentiment, use_sentiment=use_sentiment)


        if len(data) < 1:
            continue
        ages_includes.append(age)
        vectors = get_embeddings(data)

        age_utterances.append(data)
        age_vectors.append(vectors)

        all_utterances += data

        if age == 1 or first_age:
            compiled_vectors = vectors
            first_age  = False
        else:
            compiled_vectors = np.vstack((compiled_vectors, vectors))




    file_name = f'data/dimensionality_reduction/childes_{identity}_{category}_{sentiment}'
    best_k, labels, model, vectors, pca_model, dim = train_clustering_model(model_name, compiled_vectors,file_name, constant_dim, dim_reduction=dim)
    print(best_k)
    compressed_vectors = prepare_visualization(age_vectors)

    for i, age in enumerate(ages_includes):
        data = age_utterances[i]
        v = age_vectors[i]
        age_compressed_vectors = compressed_vectors[i]

        age_labels = model.predict(pca_model.transform(v))

        if model_name == 'GMM':
            probs = model.get_probs(pca_model.transform(v))

            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=True)

        else:
            probs = None
            labels_utterances = assign_labels(age_labels, data, age_compressed_vectors, probs, has_probs=False)

        if use_sentiment:
            pickle.dump((labels_utterances, probs, best_k, model),
                        open(
                            os.path.join('data', model_name, identity, f'{category}-{sentiment}', f'{age}_all_dimension_reduced_{dim}.pkl'),
                            'wb'))

        else:

            pickle.dump((labels_utterances, probs, best_k, model),
                        open(os.path.join('data', model_name, identity, category, f'{age}_all_dimension_reduced_{dim}.pkl'), 'wb'))


def category_cluster_stories(stories_data: pd.DataFrame, category, model_name, dim_reduction: int = 200) -> None:
    all_utterances = []
    age_utterances = []
    age_vectors = []
    quantiles = {i * 0.1: stories_data["flesch_kincaid"].quantile(i * 0.1) for i in range(1, 11)}


    for i, value in enumerate(quantiles):
        if i == len(quantiles) - 1:
            break
        data = select_story_data(stories_data, quantiles[value], quantiles[(i + 2) * 0.1], category)
        vectors = get_embeddings(data)
        age_utterances.append(data)
        age_vectors.append(vectors)
        all_utterances += data

    print(np.shape(age_vectors))
    best_k, labels, model, vectors, pca_model, dim = train_clustering_model(model_name, all_utterances, file_name='',constant_dim=True,
                                                                       dim_reduction=dim_reduction)
    compressed_vectors = prepare_visualization(age_vectors)
    age_vectors = pca_model.transform(age_vectors)

    for i, age in enumerate(quantiles):
        if i == len(quantiles) - 1:
            break
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
                    open(os.path.join('data/full/', model_name, category, f'{age}_all_dimension_reduced_{dim}.pkl'), 'wb'))


def find_childes_explained_variance(childes_data: pd.DataFrame,category, sentiment, use_sentiment = False, identity = 'child',):
    all_utterances = []
    age_utterances = []
    age_vectors = []
    first_age = True
    ages = range(1, 7)
    for age in ages:
        data = select_childes(childes_data, identity, age, category, sentiment, use_sentiment )
        if len(data) < 1:
            continue

        vectors = get_embeddings(data)

        age_utterances.append(data)
        age_vectors.append(vectors)

        all_utterances += data

        if age == 1 or first_age:
            compiled_vectors = vectors
            first_age = False
        else:
            compiled_vectors = np.vstack((compiled_vectors, vectors))

    # embeddings = get_embeddings(all_utterances)

    find_explained_variance(compiled_vectors, f'data/dimensionality_reduction/childes_{identity}_{category}_{sentiment}')


def find_childes_explained_variance_generic_terms(childes_data: pd.DataFrame,generic_terms, identity = 'child',):
    all_utterances = []
    age_utterances = []
    age_vectors = []
    first_age = True
    ages = range(1, 7)

    for age in ages:


        data = select_childes_with_terms(childes_data, identity, age, generic_terms)
        if len(data) < 1:
            continue

        vectors = get_embeddings(data)

        age_utterances.append(data)
        age_vectors.append(vectors)

        all_utterances += data

        if age == 1 or first_age:
            compiled_vectors = vectors
            first_age = False
        else:
            compiled_vectors = np.vstack((compiled_vectors, vectors))

    # embeddings = get_embeddings(all_utterances)

    find_explained_variance(compiled_vectors, f'data/dimensionality_reduction/childes_{identity}_{generic_terms}')


def generic_clusters():
    df = pickle.load(open(("./data/moral_df_context_general_mfd.p"), 'rb'))



    #Clustering
    terms = ['good','bad','wrong','correct']
    for t in terms:
        find_childes_explained_variance_generic_terms(df, [t],'child')

        find_childes_explained_variance_generic_terms(df,[t], 'parent')
        for m in models:



                category_cluster_generic(df, [t],model_name= m,identity='parent',  reduce=False,
                                 constant_dim=False,)



                category_cluster_generic(df, [t],model_name= m,identity='child',  reduce=False,
                                 constant_dim=False,)



def cluster_mfd2():

    df = pickle.load(open('data/moral_df_context_all-lemmatized.p', 'rb'))

    #storing the PCA data

    for c in categories:
        for p in ['pos', 'neg']:
            find_childes_explained_variance(df,c,p, use_sentiment=True,identity='child')

    for c in categories:
        for p in ['pos', 'neg']:
            find_childes_explained_variance(df,c,p, use_sentiment=True,identity='parent')



    for m in models:
        for c in categories:
            for p in ['pos', 'neg']:
                category_cluster(df,c, m, identity='parent', sentiment=p, use_sentiment=True,  constant_dim = False)



    for m in models:
        for c in categories:
            for p in ['pos', 'neg']:
                category_cluster(df,c, m, identity='child', sentiment=p, use_sentiment=True, constant_dim = False)




if __name__ == '__main__':
    cluster_mfd2()


