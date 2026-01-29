from classes import *


output_dir = './data/output/teva/'
in_dir = './data/input/edz_results/'
geomorph_path = in_dir + 'network/reach_data.csv'
edz_path = in_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
teva = TEVAHandler(reaches)
teva.prep_input_data(output_dir, edzs_only=True, binned=False)