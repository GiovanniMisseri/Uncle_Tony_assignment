import pandas as pd
import argparse
import os
from Split_into_classes import Split_into_classes

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, help="specify the path to the csv file containing the students' form answers.", required=True)
parser.add_argument('--pca_dimension', type=int, help='Default=20. Specify the number of PCA dimentions to keep. Hint: run specifying visualize=True to see PCA scree plot.')
parser.add_argument('--visualize', type=str2bool, help='Default=False. Set to True to visualize plots and statistics.')
parser.add_argument('--num_clusters', type=int, help='Default=13. Specify the number of clusters to use. Hint: run specifying visualize=True to see how inertia varies changing num_clusters.')
parser.add_argument('--output_path', type=str, help='Specify the base path where outputs should be saved.')

args = parser.parse_args()

path_data = args.path_data

pca_dimension = 20
if args.pca_dimension:
    pca_dimension = args.pca_dimension

visualize = False
if args.visualize:
    visualize = args.visualize

num_clusters = 13
if args.num_clusters:
    num_clusters = args.num_clusters

split_obj = Split_into_classes(path_data = path_data, pca_dimention=pca_dimension, num_clusters=num_clusters, visualize=visualize )

assigned_class = split_obj.distribute_students_per_class()
if args.output_path:
    base_path = args.output_path
    if base_path[-1]=='/':
        out_path = base_path+'class_dist.csv'
    else:
        out_path = base_path+'/class_dist.csv'

    os.makedirs(base_path, exist_ok=True)
    pd.DataFrame(assigned_class).to_csv(out_path, index=False)
    print('Classes assignments saved  in "{}"'.format(out_path))

