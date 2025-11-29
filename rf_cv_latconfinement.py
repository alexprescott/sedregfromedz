import numpy as np
from classes import *


output_table = '/root/repositories/sedregfromedz/data/rf_cv_latconfinement.csv'
hm_dir = '/root/SRCclusters/results/EDZcode/5Nov2025/' # 8Jul2025/'
geomorph_path = hm_dir + 'network/reach_data.csv'
edz_path = hm_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
reaches.src_edz_data['w_4times_bf_to_w_bf'] = reaches.src_edz_data['w_4times_bf'] / reaches.src_edz_data['w_bf']
reaches.src_edz_data['w_6times_bf_to_w_bf'] = reaches.src_edz_data['w_6times_bf'] / reaches.src_edz_data['w_bf']
reaches.src_edz_data['w_edep_to_w_bf'] = reaches.src_edz_data['valley_confinement'] * reaches.src_edz_data['w_edep']
reaches.src_edz_data['w_edep_to_w_bf'] /= reaches.src_edz_data['w_bf']

rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(10), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)

print("Confined vs Unconfined repeated cross-validation comparison of all vs TEVA-selected EDZ featues")
print()
class1 = ['TR', 'CST',]
class2 = ['CEFD', 'DEP', 'FSTCD', 'UST',]
cols_teva = ['Ph2SedReg','w_edep_scaled','valley_confinement','vol_scaled', 'cumulative_volume', 'stdev_rhp','w_edep_to_w_bf','w_4times_bf_to_w_bf']  # 'vol','el_edap','rh_edap' 'cumulative_height', 'height'
reaches.repeat_cv_compare(class1, class2, cols_teva, output_table)
