from classes import *
from sklearn.metrics import balanced_accuracy_score


output_table = '/root/repositories/sedregfromedz/data/rf_cv_latconfinement.csv'
hm_dir = '/root/SRCclusters/results/EDZcode/10Dec2025/'
geomorph_path = hm_dir + 'network/reach_data.csv'
edz_path = hm_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)

rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(10), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)

print("Confined vs Unconfined repeated cross-validation comparison of all vs TEVA-selected EDZ featues")
print()
class1 = ['TR', 'CST',]
class2 = ['CEFD', 'DEP', 'FSTCD', 'UST',]
cols_teva = ['Ph2SedReg', 'w_edep_scaled', 'rh_pre', 'w_4timesbf_to_w_bf']
print(f'TEVA selected features: {cols_teva}')
reaches.repeat_cv_compare(class1, class2, cols_teva, output_table)
