import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class Split_into_classes:
    def __init__(self, path_data, pca_dimention=20, num_clusters=13, visualize=False):
        self.path_data = path_data
        self.pca_dimension = pca_dimention
        self.visualize = visualize
        self.num_clusters = num_clusters

    def students_score(self):
        students = pd.read_csv(self.path_data)
        return students['score']

    def pca_data(self, is_km=False):
        visualize = self.visualize
        if is_km:
            visualize = False


        students = pd.read_csv(self.path_data)
        pca = PCA(n_components=200)
        components_200_noscore = pca.fit_transform(students.iloc[:, :-1])
        dimension = self.pca_dimension

        if visualize:
            PC_values = np.arange(pca.n_components_) + 1
            plt.figure()
            plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
            plt.title('Scree Plot PCA')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            plt.show()
            var_explaned = (pca.explained_variance_ratio_[:dimension]).sum()
            print('Variance explaned by the fist {} principal components: {}'.format(dimension, var_explaned))

        return components_200_noscore[:, :dimension]

    def kmeans_label(self):

        pca_data = self.pca_data(is_km=True)
        num_clusters = self.num_clusters

        if self.visualize:

            # choose K
            sse = []
            for k in range(1, 51):
                kmeans = KMeans(n_clusters=k).fit(pca_data)
                sse.append(kmeans.inertia_)

            plt.figure()
            plt.plot(sse)
            plt.axvline(x=num_clusters, ymin=0, ymax=700, color='red', linestyle='dashed')
            plt.title('Kmeans inertia')
            plt.xlabel('Inertia value')
            plt.ylabel('Number of clusters')
            plt.show()

            # silhouette
            sil = []
            kmax = num_clusters + 10
            # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
            for k in range(2, kmax + 1):
                kmeans = KMeans(n_clusters=k).fit(pca_data)
                labels = kmeans.labels_
                sil.append(silhouette_score(pca_data, labels, metric='euclidean'))

            plt.figure()
            plt.plot(
                sil)  # guardo il silhouette score al variare del numero di cluster per vere se questa conferma la credibilita' di 13 come numero di cluster rispetto agli altri
            plt.axvline(x=num_clusters, color='red', linestyle='dashed')
            plt.title('Kmeans Silhouette')
            plt.xlabel('Silhouette score')
            plt.ylabel('Number of clusters')
            plt.show()

        kmeans = KMeans(n_clusters=num_clusters).fit(pca_data)

        labels_km = kmeans.labels_

        if self.visualize:
            unique, counts = np.unique(labels_km, return_counts=True)
            print('Distribution students per cluster')
            print(dict(zip(unique, counts)))
            print('')

        return labels_km

    def distribute_students_per_class(self):
        score_cluster = pd.DataFrame(self.pca_data())
        score_cluster['score'] = self.students_score()
        cluster_one_comp = 1
        while cluster_one_comp > 0:
            score_cluster['cluster'] = self.kmeans_label()

            cluster_one_comp = (np.unique(score_cluster['cluster'], return_counts=True)[1] == 1).sum()

        t_test_list = []

        if self.visualize:
            print(
                'List of score per number of classes and relative number of significatively different variables wr to the population')
        min_t_score = 99999
        for num_class in range(2, 11):
            l1 = list((np.arange(num_class) + 1) / num_class)
            l1.insert(0, 0)
            l2 = list(np.arange(num_class) + 1)
            list_quantiles = []

            for clus_id in range(self.num_clusters):
                rank_score = score_cluster[score_cluster.cluster == clus_id].score.rank(method='first')
                quantiles = pd.qcut(rank_score, l1, labels=l2)
                list_quantiles.append(quantiles)
            score_cluster['class_tmp'] = pd.concat(list_quantiles)

            mean_df = score_cluster.groupby(['class_tmp']).mean().iloc[:, :self.pca_dimension]
            std_df = score_cluster.groupby(['class_tmp']).std().iloc[:, :self.pca_dimension]
            n_df = (score_cluster.groupby(['class_tmp']).count()).iloc[:, :self.pca_dimension]
            SE_df = std_df / np.sqrt(n_df)
            t_test_df = (mean_df / SE_df).iloc[:, :self.pca_dimension]
            t_score_abs_sum = np.abs(t_test_df).sum().sum() / num_class
            if self.visualize:
                print(t_score_abs_sum, ((np.abs(t_test_df) > 1.96).sum().sum()))

            t_test_list.append(t_score_abs_sum)

            if t_score_abs_sum < min_t_score:
                min_t_score = t_score_abs_sum
                score_cluster['class'] = pd.concat(list_quantiles)
                num_class_sel = num_class

        score_cluster = score_cluster.drop(['class_tmp'], axis=1)
        if self.visualize:
            print('')

            print('Selected number of classes: {}'.format(num_class_sel))
            print('')

            avg_score_class = score_cluster.groupby(['class'])['score'].mean()
            print('Avg score per class')
            print(avg_score_class)
            print('')

            num_students_per_class = score_cluster.groupby(['class'])['score'].count()
            print('Number of students per class')
            print(num_students_per_class)
            print('')

        return score_cluster['class']



