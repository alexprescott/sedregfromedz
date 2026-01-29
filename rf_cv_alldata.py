from classes import *
from sklearn.metrics import balanced_accuracy_score


output_table = './data/output/rf_cv_alldata.csv'
in_dir = './data/input/'
geomorph_path = in_dir + 'network/reach_data.csv'
edz_path = in_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(5), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)

print("All data repeated cross-validation comparison of all vs TEVA-selected EDZ featues")
print()
cols_teva = ['Ph2SedReg', 'rh_edap', 'cumulative_height', 'valley_confinement', 'w_3timesbf_to_w_bf', 'el_edap_scaled', 'w_edap_scaled','rhp_pre', 'edz_count', 'wtod_bf', 'ssp_3times_bf', 'Ave_Rh', 'min_loc_ratio', 'slope']
print(f'TEVA selected features: {cols_teva}')
reaches.repeat_cv_compare_alldata(cols_teva, output_table)
