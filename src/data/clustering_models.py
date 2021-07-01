from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np


class ClusteringModel():

    def __init__(self):

        self.ks = range(2, 11)

    def train(self, data, k):
        pass

    def find_k(self, data):
        ks_scores = {}
        ks_lables = {}
        ks_models = {}
        for k in self.ks:
            if len(data) - 1 < k:
                continue
            labels, model = self.train(data, k)

            score = silhouette_score(data,labels)

            ks_scores[k] = score
            ks_lables[k] = labels
            ks_models[k] = model

        best_k = max(ks_scores.items(), key = lambda x : x[1])

        self.model = ks_models[best_k[0]]
        return best_k, ks_lables[best_k[0]], ks_models[best_k[0]]


    def predict(self, data):
        assignment = self.model.predict(data)
        return assignment


class KmeansModel(ClusteringModel):

    name = 'Kmeans'
    def train(self, data: np.array, k : int):
        clustering_model = cluster.KMeans(n_clusters=k)
        clustering_model.fit(data)
        assignment = clustering_model.labels_


        return assignment, clustering_model


class GMM(ClusteringModel):

    name = 'GMM'
    def train(self, data: np.array, k : int):

        gmm_model = GaussianMixture(n_components=k, random_state=0)
        gmm_model.fit(data)
        assignment = gmm_model.predict(data)


        return assignment, gmm_model



    def get_probs(self, data):
        probs = self.model.predict_proba(data)
        return probs







