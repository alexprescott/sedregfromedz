from classes import *


output_dir = '/root/repositories/sedregfromedz/data/teva/'
hm_dir = '/root/SRCclusters/results/EDZcode/5Nov2025/' # 8Jul2025/'
geomorph_path = hm_dir + 'network/reach_data.csv'
edz_path = hm_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
teva = TEVAHandler(reaches)
teva.prep_input_data(output_dir, edzs_only=True, binned=False)