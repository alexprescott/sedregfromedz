from classes import *
from sklearn.metrics import balanced_accuracy_score

hm_dir = '/root/SRCclusters/results/EDZcode/10Dec2025/'  # 5Nov2025/' # 8Jul2025/'
geomorph_path = hm_dir + 'network/reach_data.csv'
edz_path = hm_dir + 'analysis/data.csv'
reach_avg_dir = hm_dir + 'geometry/'
out_dir = "./figs/"


reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(10), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)
reaches.load_reach_avg_profiles(reach_avg_dir)

fig = reaches.plot_edz_feature_descriptions()
fig.savefig(out_dir + 'dem_feat_desc.tif', dpi=300, bbox_inches='tight')

fig = reaches.plot_reconstructed_xsecs_paper()
fig.savefig(out_dir + 'xsecs.tif', dpi=300, bbox_inches='tight')

#fig = reaches.plot_sankey_diagram()
#fig.savefig(out_dir + 'sankey.tif', dpi=300, bbox_inches='tight')

fig = reaches.plot_boxplots_paper()
fig.savefig(out_dir + 'boxplots.tif', dpi=300, bbox_inches='tight')

