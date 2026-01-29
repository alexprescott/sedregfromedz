from classes import *
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
from sklearn.metrics import balanced_accuracy_score

in_dir = './data/input/'
geomorph_path = in_dir + 'network/reach_data.csv'
edz_path = in_dir + 'analysis/data.csv'
reach_avg_dir = in_dir + 'geometry/'
out_dir = "./figs/"


reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
rf = RandomForestCVHandler(rf_kwargs = {'n_estimators': 1000, 'oob_score': balanced_accuracy_score, 'class_weight': 'balanced', 'max_depth': 4,}, cv_kwargs={'cv': StratifiedKFold(10), 'return_estimator': True, 'scoring': 'balanced_accuracy'})
reaches.attach_rfhandler(rf)
reaches.load_reach_avg_profiles(reach_avg_dir)
pi_kwargs = {'n_repeats': 500, 'n_jobs': 10, 'scoring': 'balanced_accuracy'}


# First, permutation importances for the main paper
print()
print("Main figure")
edz_fields_map1 = {
    'w_edep_scaled': 'Scaled width at EDZ exit',
    'rh_pre': 'Mean Rh below EDZ',
    'w_4timesbf_to_w_bf': '4-times bankfull width ratio',
}
edz_features1 = list(edz_fields_map1.keys())
X1 = reaches.src_edz_data[edz_features1]
y1 = reaches.src_edz_data['Ph2SedReg'].isin(['TR','CST']).astype(np.int8)
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X1, y1.to_numpy().ravel())
pi_results1 = reaches.rf.permutation_importances(clf, X1, y1, pi_kwargs=pi_kwargs)

edz_fields_map2 = {
    'cumulative_height': 'Total EDZ stage range', 
    'rhp_post_stdev': 'Std. dev. of Rh\' above EDZ',
    'el_edap_scaled': 'EDZ access stage',
    'Ave_Rh': 'Mean Rh',
    'rh_edap': 'Rh at EDZ access',
    'cumulative_volume': 'Total EDZ diagnostic size',
}
edz_features2 = list(edz_fields_map2.keys())
chosen_regs = ['CEFD','FSTCD','DEP','UST']
mask = reaches.src_edz_data['Ph2SedReg'].isin(chosen_regs)
X2 = reaches.src_edz_data[edz_features2].loc[mask]
y2 = reaches.src_edz_data.Ph2SedReg.loc[mask].isin(['CEFD', 'DEP'])
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X2, y2.to_numpy().ravel())
pi_results2 = reaches.rf.permutation_importances(clf, X2, y2, pi_kwargs=pi_kwargs)

edz_fields_map3 = {
    "wtod_bf": "Bankfull width to depth",
    'rhp_pre': 'Mean Rh\' below EDZ',
    'w_edep_scaled': 'Scaled width at EDZ exit',
    'rh_pre': 'Mean Rh below EDZ',
}
edz_features3 = list(edz_fields_map3.keys())
chosen_regs = ['FSTCD','UST']
mask = reaches.src_edz_data['Ph2SedReg'].isin(chosen_regs)
X3 = reaches.src_edz_data[edz_features3].loc[mask]
y3 = reaches.src_edz_data.Ph2SedReg.loc[mask].isin(['FSTCD'])
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X3, y3.to_numpy().ravel())
pi_results3 = reaches.rf.permutation_importances(clf, X3, y3, pi_kwargs=pi_kwargs)

fig,axs = plt.subplots(3,1, figsize=(3.,4.5), layout='constrained', height_ratios=[len(edz_features1), len(edz_features2), len(edz_features3)])
pirs=[pi_results1, pi_results2, pi_results3]
titles=['Lateral confinement', 'Floodplain (dis)connection', 'FSTCD vs. UST']
field_maps=[edz_fields_map1, edz_fields_map2, edz_fields_map3]

for pir,ax,title,fmap in zip(pirs,axs,titles,field_maps):
    means = pir.mean(axis=0)
    stds = pir.std(axis=0)
    b = means.plot.barh(xerr=stds, ax=ax, capsize=0, width=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('')
    ax.set_xlim([0, ax.get_xlim()[1]])  # axvline(x=0, color="k",)
    yticklabs = ax.get_yticklabels()
    yticklabels = [fmap[k.get_text()] for k in yticklabs]
    ax.set_yticklabels(yticklabels)
axs[0].set_xticks([0, 0.2, 0.4])
axs[-1].set_xlabel("Mean decrease in accuracy", fontsize=8)

for ax,letter in zip(axs,['a','b','c']):
    ax.text(-1.2, 1.125, f'({letter})', ha='left', va='center', transform=ax.transAxes)

fig.savefig(out_dir + 'permutation_importances.tif', dpi=300, bbox_inches='tight')



# Now make the plot for the supplement
print()
print("Supplement figure")
edz_fields_map4 = {
    'Ave_Rh': 'Mean Rh',
    'valley_confinement': 'EDZ relative width',
    'w_2timesbf': 'Width at 2-times bankfull stage',
    'w_3timesbf_to_w_bf': '3-times bankfull width ratio',
}
edz_features4 = list(edz_fields_map4.keys())
chosen_regs = ['TR','CST',]
mask = reaches.src_edz_data['Ph2SedReg'].isin(chosen_regs)
X4 = reaches.src_edz_data[edz_features4].loc[mask]
y4 = reaches.src_edz_data.Ph2SedReg.loc[mask].isin(['TR',])
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X4, y4.to_numpy().ravel())
pi_results4 = reaches.rf.permutation_importances(clf, X4, y4, pi_kwargs=pi_kwargs)

edz_fields_map5 = {
    'slope': 'Channel slope',
    'ssp_bf': 'Bankfull specific stream power'
}
edz_features5 = list(edz_fields_map5.keys())
chosen_regs = ['CEFD','DEP',]
mask = reaches.src_edz_data['Ph2SedReg'].isin(chosen_regs)
X5 = reaches.src_edz_data[edz_features5].loc[mask]
y5 = reaches.src_edz_data.Ph2SedReg.loc[mask].isin(['CEFD',])
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X5, y5.to_numpy().ravel())
pi_results5 = reaches.rf.permutation_importances(clf, X5, y5, pi_kwargs=pi_kwargs)

edz_fields_map6 = {
    'rh_edap': 'Rh at EDZ access',
    'cumulative_height': 'Total EDZ stage range',
    'valley_confinement': 'EDZ relative width',
    'w_3timesbf_to_w_bf': 'Widths ratio, 3x BF to BF',
    'el_edap_scaled': 'EDZ access stage',
    'w_edap_scaled': 'Scaled width at EDZ access',
    'rhp_pre': 'Mean Rh\' below the EDZ',
    'edz_count': 'EDZ count',
    'wtod_bf': 'BF width-to-depth ratio',
    'ssp_3times_bf': 'SSP at 3x BF',
    'Ave_Rh': 'Mean Rh',
    'min_loc_ratio': 'Fractional stage of\nmax. lateral expansion',
    'slope': 'Channel slope'
}
edz_features6 = list(edz_fields_map6.keys())
X6 = reaches.src_edz_data[edz_features6]
labs = reaches.src_edz_data.Ph2SedReg.unique()
nums = np.arange(len(labs))
labs_nums_dict = dict(zip(labs, nums))
y6 = reaches.src_edz_data.Ph2SedReg.copy()
y6 = y6.map(labs_nums_dict).astype(np.int8)
clf = RandomForestClassifier(**reaches.rf.rf_kwargs)
clf.fit(X6, y6.to_numpy().ravel())
pi_results6 = reaches.rf.permutation_importances(clf, X6, y6, pi_kwargs=pi_kwargs)

fig = plt.figure(layout='constrained', figsize=(6.4,3.2))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[:,1])
axs = [ax1, ax2, ax3]

pirs=[pi_results4, pi_results5, pi_results6]
titles=['(a) TR vs. CST', '(b) CEFD vs. DEP', '(c) All reaches']
field_maps=[edz_fields_map4, edz_fields_map5, edz_fields_map6]
for pir,ax,title,fmap in zip(pirs,axs,titles,field_maps):
    means = pir.mean(axis=0)
    stds = pir.std(axis=0)
    b = means.plot.barh(xerr=stds, ax=ax, capsize=0, width=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('')
    ax.set_xlim([0, ax.get_xlim()[1]])  # axvline(x=0, color="k",)
    yticklabs = ax.get_yticklabels()
    yticklabels = [fmap[k.get_text()] for k in yticklabs]
    ax.set_yticklabels(yticklabels)

for ax in axs[1:]:
    ax.set_xlabel("Decrease in accuracy", fontsize=8)

fig.savefig(out_dir + 'permutation_importances_supp.tif', dpi=300, bbox_inches='tight')