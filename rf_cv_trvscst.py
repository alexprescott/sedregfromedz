from classes import *
from sklearn.metrics import balanced_accuracy_score


output_table = '/root/repositories/sedregfromedz/data/rf_cv_trvscst.csv'
hm_dir = '/root/SRCclusters/results/EDZcode/10Dec2025/'
geomorph_path = hm_dir + 'network/reach_data.csv'
edz_path = hm_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(8), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)

print("TR vs CST repeated cross-validation comparison of all vs TEVA-selected EDZ featues")
print()
class1 = ['TR',]
class2 = ['CST',]
cols_teva = ['Ph2SedReg','Ave_Rh','valley_confinement','w_2timesbf','w_3timesbf_to_w_bf']
print(f'TEVA selected features: {cols_teva}')
reaches.repeat_cv_compare(class1, class2, cols_teva, output_table)
