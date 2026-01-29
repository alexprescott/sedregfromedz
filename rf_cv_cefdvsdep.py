from classes import *
from sklearn.metrics import balanced_accuracy_score


output_table = './data/output/rf_cv_cefdvsdep.csv'
in_dir = './data/input/edz_results/'
geomorph_path = in_dir + 'network/reach_data.csv'
edz_path = in_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(5), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)

print("CEFD vs DEP repeated cross-validation comparison of all vs TEVA-selected EDZ featues")
print()
class1 = ['CEFD',]
class2 = ['DEP',]
cols_teva = ['Ph2SedReg','slope', 'ssp_bf',]
print(f'TEVA selected features: {cols_teva}')
reaches.repeat_cv_compare(class1, class2, cols_teva, output_table)
