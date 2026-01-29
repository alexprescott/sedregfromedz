import numpy as np
from classes import *
from matplotlib.gridspec import GridSpec


plt.rcParams.update({'legend.fontsize': 8})

in_dir = './data/input/edz_results/'
geomorph_path = in_dir + 'network/reach_data.csv'
edz_path = in_dir + 'analysis/data.csv'
reaches = ReachData(geomorphic_data_path=geomorph_path, edz_data_path=edz_path)
out_dir = "./figs/"


# try putting it all in one figure, 3 rows:
# 1. Vertical (dis)connectivity
# 2.i. Confinement
# 2.ii. FSTCD vs. UST
# 3.i CEFD vs. DEP
# 3.ii TR vs. CST

fig = plt.figure(layout='constrained', figsize=(6.5,8.75))
gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])

# Confinement
edz_fields_map = {
    'w_edep_scaled': 'Scaled width at EDZ exit',
    'rh_pre': 'Mean Rh below EDZ',
    'w_4timesbf_to_w_bf': '4-times bankfull width ratio',
}
class1 = ['TR', 'CST']
class2 = ['CEFD', 'DEP', 'FSTCD', 'UST']
label1 = 'Confined'
label2 = 'Unconfined'
colors = [reaches.hexcolors_groups[label1], reaches.hexcolors_groups[label2]]
ax = reaches.plot_boxplots_supplement(edz_fields_map, class1, class2, label1, label2, ax2, colors=colors, log_scale=True)
ax.set_title('(b) Lateral confinement', fontsize=12)


# Unconfined, vertically connected vs. disconnected
edz_fields_map = {
    'cumulative_height': 'Total EDZ stage range', 
    'rhp_post_stdev': 'Std. dev. of Rh\' above EDZ',
    'el_edap_scaled': 'EDZ Access Stage',
    'Ave_Rh': 'Mean Rh',
    'rh_edap': 'Rh at EDZ access',
    'cumulative_volume': 'Total EDZ diagnostic size',
}
class1 = ['CEFD', 'DEP']
class2 = ['FSTCD', 'UST']
label1 = 'Connected'
label2 = 'Disconnected'
colors = [reaches.hexcolors_groups[label1], reaches.hexcolors_groups[label2]]
ax = reaches.plot_boxplots_supplement(edz_fields_map, class1, class2, label1, label2, ax1, colors=colors)
ax.set_title('(a) Floodplain (dis)connectivity', fontsize=12)


# FSTCD vs. UST
edz_fields_map = {
    'rhp_pre': 'Mean Rh\' below EDZ',
    'wtod_bf': 'Bankfull width to depth',
    'w_edep_scaled': 'Scaled Width at EDZ Exit',
    'rh_pre': 'Mean Rh below EDZ',
}
class1 = ['FSTCD',]
class2 = ['UST',]
label1 = 'FSTCD'
label2 = 'UST'
colors = [reaches.hexcolors_U2021[label1], reaches.hexcolors_U2021[label2]]
ax = reaches.plot_boxplots_supplement(edz_fields_map, class1, class2, label1, label2, ax3, colors=colors, log_scale=True)
ax.set_title('(c) FSTCD vs. UST', fontsize=12)

# CEFD vs. DEP
edz_fields_map = {
    'slope': 'Channel slope',
    'ssp_bf': 'Bankfull specific stream power'
}
class1 = ['CEFD',]
class2 = ['DEP',]
label1 = 'CEFD'
label2 = 'DEP'
colors = [reaches.hexcolors_U2021[label1], reaches.hexcolors_U2021[label2]]
ax = reaches.plot_boxplots_supplement(edz_fields_map, class1, class2, label1, label2, ax4, colors=colors, log_scale=True)
ax.set_title('(d) CEFD vs. DEP', fontsize=12)

# TR vs. CST
edz_fields_map = {
    'Ave_Rh': 'Mean Rh',
    'valley_confinement': 'EDZ relative width',
    'w_3timesbf_to_w_bf': '3-times bankfull width ratio',
    'w_2timesbf': 'Width at 2-times bankfull stage',
}
class1 = ['TR',]
class2 = ['CST',]
label1 = 'TR'
label2 = 'CST'
colors = [reaches.hexcolors_U2021[label1], reaches.hexcolors_U2021[label2]]
ax = reaches.plot_boxplots_supplement(edz_fields_map, class1, class2, label1, label2, ax5, colors=colors, log_scale=True)
ax.set_title('(e) TR vs. CST', fontsize=12)

fig.savefig(out_dir + 'supp_panel_boxplots.tif', dpi=300, bbox_inches='tight')