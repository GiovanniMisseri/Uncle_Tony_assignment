import pandas as pd
import unittest
from random import sample
from Split_into_classes import Split_into_classes
import argparse
import numpy as np

class TestSplit_into_classes_algo(unittest.TestCase):

    def test_CorrectNumOutput(self):
        i = 0
        while i < 10:
            sel_pca_dimension = sample(range(2, 200), 1)[0]
            sel_num_clusters = sample(range(2, 50), 1)[0]
            algo = Split_into_classes(path_data=self.path_data, pca_dimention=sel_pca_dimension, num_clusters=sel_num_clusters)
            class_dist = algo.distribute_students_per_class()
            students_score = algo.students_score()
            self.assertEqual(len(class_dist), len(students_score),
                             'Input number of students different from output students in classes')

            i = i + 1

    def test_avg_score(self):
        i = 0
        while i < 10:
            sel_pca_dimension = sample(range(2, 200), 1)[0]
            sel_num_clusters = sample(range(2, 50), 1)[0]

            algo = Split_into_classes(path_data=self.path_data, pca_dimention=sel_pca_dimension, num_clusters=sel_num_clusters)
            class_dist = algo.distribute_students_per_class()
            students_score = algo.students_score()

            list_avg_score = list(pd.concat([students_score, class_dist], axis=1).groupby(['class']).mean().score)
            self.assertEqual(list_avg_score, sorted(list_avg_score),
                             'Not distributing students in classes ordered by skill score')

            i = i + 1

    def test_classes_size(self):
        i = 0
        while i < 10:
            sel_pca_dimension = sample(range(2, 200), 1)[0]
            sel_num_clusters = sample(range(2, 50), 1)[0]

            algo = Split_into_classes(path_data=self.path_data, pca_dimention=sel_pca_dimension, num_clusters=sel_num_clusters)
            class_dist = algo.distribute_students_per_class()
            class_size = class_dist.value_counts()
            theo_size = len(class_dist) / len(class_size)
            abs_diff = np.abs(class_size / theo_size - 1)
            self.assertTrue(sum(abs_diff < 0.2) == len(class_size),
                            'Not distributing students in classes similar in size')

            i = i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, help="specify the path to the csv file containing the students' form answers.", required=True)
    args = parser.parse_args()
    TestSplit_into_classes_algo.path_data = args.path_data
    unittest.main(argv=['first-arg-is-ignored'], exit=False)