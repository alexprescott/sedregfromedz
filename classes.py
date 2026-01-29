import colorsys
import logging
import matplotlib.colors as mc
import matplotlib.lines as lines
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import textwrap
import teva
from matplotlib.colors import LinearSegmentedColormap
from sankeyflow import Sankey
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kstest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.inspection import permutation_importance

plt.rcParams.update({'font.size': 8}) 


class ReachData:

    def __init__(self, geomorphic_data_path, edz_data_path, drop_unconfined_edz_nan=True):
        df1 = pd.read_csv(geomorphic_data_path)
        df2 = pd.read_csv(edz_data_path)
        df = df1.merge(df2, on='SRC_ID', suffixes=(None, '_y')) # note that columns length, slope, and wbody are duplicated in both dataframes, hence we specify None and _y suffixes
        df = df.drop(columns=df.columns[df.columns.str.contains('_y')])  # the duplicated fields are exactly equal, so we drop the _y columns
        df['SRC_ID'] = df['SRC_ID'].astype(np.int64).astype(str)  # avoids some key issues later
        df1['SRC_ID'] = df1['SRC_ID'].astype(np.int64).astype(str)
        df2['SRC_ID'] = df2['SRC_ID'].astype(np.int64).astype(str)
        df.set_index('SRC_ID', inplace=True)
        df1.set_index('SRC_ID', inplace=True)
        df2.set_index('SRC_ID', inplace=True)

        print(f"Input data shape: {df.shape}")
        print()
        if drop_unconfined_edz_nan:
            hand_cliff_reaches = df[(df['el_edap_scaled'].isnull()) & (df['Ph2SedReg'].isin(['CEFD','FSTCD','DEP','UST']))].index
            print(f"Unconfined reaches without an extracted EDZ:")
            print(df.loc[hand_cliff_reaches, ['Segment', 'Ph2SedReg']])
            print()
            df1 = df1.drop(index=hand_cliff_reaches)
            df2 = df2.drop(index=hand_cliff_reaches)
            df = df.drop(index=hand_cliff_reaches)
            print(f"Shape after dropping unconfined NaNs: {df.shape}")
            print(f'{df.Ph2SedReg.value_counts()}')
            print()
        
        self.geomorphic_data = df1
        self.edz_data = df2
        self.src_edz_data = df

        self.hexcolors_U2021 = {
            'CEFD': '#019d71b3',
            'CST': '#fce900b3',
            'DEP': '#f798b6b3',
            'FSTCD': '#fe0000b3',
            'TR': '#0b68f0b3',
            'UST': '#ff9901b3',
        }

        self.hexcolors_groups = {
            'Confined': '#808080',
            'Unconfined': '#d3d3d3',
            'Connected': '#00ff78b3',
            'Disconnected': '#FF0087b3',
        }
    
    def attach_rfhandler(self, rf):
        self.rf = rf

    def load_reach_avg_profiles(self, dir):
        self.area_data = pd.read_csv(os.path.join(dir, 'area.csv'))
        self.el_data = pd.read_csv(os.path.join(dir, 'el.csv'))
        self.el_scaled_data = pd.read_csv(os.path.join(dir, 'el_scaled.csv'))
        self.rh_data = pd.read_csv(os.path.join(dir, 'rh.csv'))
        self.rh_prime_data = pd.read_csv(os.path.join(dir, 'rh_prime.csv'))
        self.vol_data = pd.read_csv(os.path.join(dir, 'vol.csv'))

        # clean Rh prime data
        self.rh_prime_data.iloc[-1] = self.rh_prime_data.iloc[-2]
        self.rh_prime_data[self.rh_prime_data < -3.0] = -3.0
        self.rh_prime_data = self.rh_prime_data.dropna(axis=1)

    def plot_reconstructed_xsecs_paper(self):

        regs = [['TR'], ['CEFD'], ['DEP'], ['CST'], ['UST'], ['FSTCD'],]
        titles = [x[0] for x in regs]
        n_elems = [(self.src_edz_data.Ph2SedReg.isin(x) > 0).sum() for x in regs]
        fig = plt.figure(figsize=(6.,2), constrained_layout=False, dpi=300)
        fig_widths = [5, 5, 5, 1]
        fig_heights = [1, 3, 3]
        gs = fig.add_gridspec(ncols=4, nrows=3, width_ratios=fig_widths, height_ratios=fig_heights, wspace=0.012, hspace=0.03)
        axs = []
        axs.append(fig.add_subplot(gs[0, :-1]))
        axs.append(fig.add_subplot(gs[1, 0]))
        axs.append(fig.add_subplot(gs[1, 1]))
        axs.append(fig.add_subplot(gs[1, 2]))
        axs.append(fig.add_subplot(gs[2, 0]))
        axs.append(fig.add_subplot(gs[2, 1]))
        axs.append(fig.add_subplot(gs[2, 2]))
        axs.append(fig.add_subplot(gs[:, 3]))

        reach_data = self.src_edz_data
        for reg,ax,title,n in zip(regs,axs[1:-1],titles,n_elems):
            valid_reaches = set(reach_data.index[reach_data.Ph2SedReg.isin(reg)])
            valid_reaches = sorted(valid_reaches)
            all_els = np.zeros((len(valid_reaches), 2*self.el_scaled_data.shape[0]))
            all_widths = np.zeros((len(valid_reaches), 2*self.el_scaled_data.shape[0]))
            all_edzas = np.zeros(len(valid_reaches))
            all_edzes = np.zeros(len(valid_reaches))
            all_relwidths = np.zeros(len(valid_reaches))
            all_partitions = np.zeros(len(valid_reaches), )
            for i,reach in enumerate(list(valid_reaches)):
                tmp_meta = reach_data.loc[reach]
                #tmp_meta_edzs = analysis_data.loc[reach]
                tmp_el_scaled = self.el_scaled_data[reach].to_numpy()
                tmp_area = self.area_data[reach].to_numpy() / tmp_meta['length']
                tmp_ph2sedreg = tmp_meta["Ph2SedReg"]

                tmp_rh = self.rh_data[reach].to_numpy()
                tmp_rh_prime = self.rh_prime_data[reach].to_numpy()

                # add_geometry(tmp_el_scaled, tmp_area, tmp_rh, tmp_rh_prime, ave)
                # add_geometry(self, el, width, rh, rhp, ave)
                width = tmp_area
                el = tmp_el_scaled

                width = width / 2
                width = np.append(-width[::-1], width)
                #width = width - min(width)
                #width /= width.max()
                #width -= 0.5 # center on 0
                # width = width / (2.44 * (self.da ** 0.34))
                section_el = np.append(el[::-1], el)
                all_els[i,:] = section_el
                all_widths[i,:] = width
                all_edzas[i] = tmp_meta["el_edap"]
                all_edzes[i] = tmp_meta["el_edep"]
                all_relwidths[i] = tmp_meta["valley_confinement"]
                color = self.hexcolors_U2021[tmp_ph2sedreg]
                _ = ax.plot(width, section_el, lw=1, alpha=0.7, color=color,)
            x = np.mean(all_widths, axis=0)
            y = np.mean(all_els, axis=0)
            _ = ax.plot(x, y, 'k-', lw=1, zorder=1000)
            idx_l = np.argmin(np.abs(y[:len(y)//2]-1.0))
            idx_r = (len(y)//2) + np.argmin(np.abs(y[len(y)//2:]-1.0))
            #y1 = np.nanmedian(all_edzas) * np.ones(x.size)
            #y2 = np.nanmedian(all_edzes) * np.ones(x.size)
            #_ = ax.fill_between(x, y1, y2, color='b', alpha=0.5)
            #ax.set_xlabel(r'Normalized width (m m$^{-1}$)', fontsize=8)
            #ax.set_ylabel(r'Scaled stage (m m$^{-1}$)', fontsize=8)
            ax.hlines(1.0, x[idx_l], x[idx_r], colors='dodgerblue', lw=1.0,)
            ax.add_patch(mpatches.Polygon([[0,1.03],[-2,1.5],[2,1.5]], closed=True, facecolor='dodgerblue', edgecolor='k', lw=0.5))
            text = ax.text(-49.5, 5.9, title, fontsize=10, weight='bold', ha='left', va='top', bbox=dict(boxstyle='square,pad=0', fc='white', ec='none', alpha=0.8), zorder=1001)
            text = ax.annotate(f' (n={n})', xycoords=text, xy=(1,0), fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='square,pad=0', fc='white', ec='none', alpha=0.8), zorder=1002)
            #ax.axhline(1.0, ls=':', color='k')
            ax.set_xlim([-50,50]) #ax.set_xlim([-0.5,0.5])
            ax.set_ylim([0.,6.])
            ax.set_xticks([])  # ax.set_xticks([-50, 0, 50], labels=['', '', ''])  # ax.set_xticks([-50, 0, 50], labels=['-50', '0', '50'], fontsize=8)
            ax.set_yticks([])  # ax.set_yticks([0., 3., 6.], labels=['', '', ''])  # ax.set_yticks([0., 3., 6.], labels=['0', '', '6'], fontsize=8)
            ax.set_facecolor('lightgrey')
            if reg in [['TR',], ['CST',]]:
                ax.set_facecolor('grey')

        # add text and arrows with gradients; https://stackoverflow.com/questions/72130591/fill-polygon-with-vertical-gradient
        cmap_wk = mc.LinearSegmentedColormap.from_list('white_to_black', ['white', 'black'])
        cmap_bp = mc.LinearSegmentedColormap.from_list('blue_to_pink', [self.hexcolors_groups['Connected'], self.hexcolors_groups['Disconnected']])
        grad = np.atleast_2d(np.linspace(0,1,256))

        # top axis
        ax = axs[0]
        ax.text(0.06,0.55, "Supply-limited", fontsize=8, ha='left')
        ax.text(0.94,0.55, "Transport-limited", fontsize=8, ha='right')
        img0 = ax.imshow(np.flip(grad), extent=[0, 0.5, -1, 2], interpolation='nearest', aspect='auto', cmap=cmap_wk)
        img1 = ax.imshow(grad, extent=[0.5, 1, -1, 2], interpolation='nearest', aspect='auto', cmap=cmap_wk)
        arrow0 = mpatches.FancyArrowPatch((0.5,0.4), (0,0.4), mutation_scale=20, clip_on=False, facecolor='none', edgecolor='none')
        arrow1 = mpatches.FancyArrowPatch((0.5,0.4), (1,0.4), mutation_scale=20, clip_on=False, facecolor='none', edgecolor='none')
        ax.add_patch(arrow0)
        ax.add_patch(arrow1)
        img0.set_clip_path(arrow0)
        img1.set_clip_path(arrow1)
        ax.set_xlim([0,1])
        ax.set_ylim([-0.1,0.9])
        ax.set_axis_off()

        # right-most axis
        ax = axs[-1]
        ax.text(0.25, -0.05, "Floodplain disconnectivity", fontsize=8, va="bottom", rotation=90)
        img = ax.imshow(grad.T, extent=[-0.5, 0.5, -0.1, 0.8], interpolation='nearest', aspect='auto', cmap=cmap_bp)
        arrow = mpatches.FancyArrowPatch((0.0,0.8), (0.0,-0.1), mutation_scale=20, clip_on=False, facecolor='none', edgecolor='none')
        ax.add_patch(arrow)
        img.set_clip_path(arrow)
        ax.set_xlim([-0.5,0.5 ])
        ax.set_ylim([-0.1,1])
        ax.set_axis_off()


        # manually add x-tick labels to bottom plots for custom alignment
        for ax in axs[4:-1]:
            ax.set_xticks([-50, 0, 50], labels=['', '', ''])
            ax.text(-50, -1.5, "-50", fontsize=8, ha="left")
            ax.text(0, -1.5, "0", fontsize=8, ha="center")
            ax.text(50, -1.5, "50", fontsize=8, ha="right")
        axs[5].text(0, -3, 'Centered width [m]', fontsize=8, ha="center")
        # manually add y-tick labels to left plots for custom alignment
        for ax in [axs[1], axs[4]]:
            ax.set_yticks([0., 3., 6.], labels=['', '', ''])
            ax.text(-59, 0, "0", fontsize=8, va='bottom')
            ax.text(-59, 6, "6", fontsize=8, va='top')
        fig.text(0.07, 0.45, r'Scaled stage [m m$^{-1}$]', fontsize=8, va="center", rotation=90)
        

        # custom make the legend
        handles = []
        labels = []
        handles.append(mpatches.Patch(facecolor='grey', edgecolor='black',))
        labels.append('Confined')
        handles.append(mpatches.Patch(facecolor='lightgrey', edgecolor='black',))
        labels.append('Unconfined')
        #handles.append(mpatches.Patch(facecolor='none', edgecolor='#332288', ls='--', lw=1.5,))
        #labels.append("Vertically connected")
        #handles.append(mpatches.Patch(facecolor='none', edgecolor='#ee6576', ls='--', lw=1.5,))
        #labels.append("Vertically disconnected")
        handles.append(lines.Line2D([0],[0], color='black',))
        labels.append('Mean cross section')
        class BankfullMarker:
            pass
        class BankfullMarkerHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x,y = handlebox.xdescent, handlebox.ydescent
                w,h = handlebox.width, handlebox.height
                w_half = w / 2
                h_half = h / 2
                w_tri = w / 8
                h_tri = h_half
                bf_tri = mpatches.Polygon([[x+w_half, y+1.2*h_half], [x+w_half-w_tri, y+h_half+h_tri], [x+w_half+w_tri, y+h_half+h_tri]], closed=True, facecolor='dodgerblue', edgecolor='black', lw=0.5, transform=handlebox.get_transform())
                bf_line = lines.Line2D([x, x+w], [y+h_half, y+h_half], lw=1, color='dodgerblue', transform=handlebox.get_transform())
                handlebox.add_artist(bf_tri)
                handlebox.add_artist(bf_line)
                return bf_tri, bf_line
        handles.append(BankfullMarker())
        labels.append('Bankfull depth')
        fig.legend(handles=handles, labels=labels, handler_map={BankfullMarker: BankfullMarkerHandler()}, loc='center', bbox_to_anchor=(0.5, -0.125), ncols=4, fontsize=8)

        fig.suptitle('Reach-averaged cross sections by sediment regime', fontsize=12)

        return fig

    def plot_sankey_diagram(self):
        n_conf = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['TR','CST'])].shape[0]
        n_unconf = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CEFD','DEP','FSTCD','UST'])].shape[0]
        n_vconn = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CEFD','DEP'])].shape[0]
        n_vdisconn = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['FSTCD','UST'])].shape[0]
        n_cst = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CST',])].shape[0]
        n_tr = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['TR',])].shape[0]
        n_cefd = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CEFD',])].shape[0]
        n_dep = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['DEP',])].shape[0]
        n_fstcd = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['FSTCD',])].shape[0]
        n_ust = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['UST',])].shape[0]

        flows = [
            ('All data', 'Confined', n_conf, {'color': self.hexcolors_groups['Confined']}),
            ('All data', 'Unconfined', n_unconf, {'color': self.hexcolors_groups['Unconfined']}),
            ('Confined', 'CST', n_cst, {'color': self.hexcolors_U2021['CST']}),
            ('Confined', 'TR', n_tr, {'color': self.hexcolors_U2021['TR']}),
            ('Unconfined', 'Vertically\nconnected', n_vconn, {'color': self.hexcolors_groups['Connected']}),
            ('Unconfined', 'Vertically\ndisconnected', n_vdisconn, {'color': self.hexcolors_groups['Disconnected']}),
            ('Vertically\nconnected', 'CEFD', n_cefd, {'color': self.hexcolors_U2021['CEFD']}),
            ('Vertically\nconnected', 'DEP', n_dep, {'color': self.hexcolors_U2021['DEP']}),
            ('Vertically\ndisconnected', 'UST', n_ust, {'color': self.hexcolors_U2021['UST']}),
            ('Vertically\ndisconnected', 'FSTCD', n_fstcd, {'color': self.hexcolors_U2021['FSTCD']}),
        ]

        my_cmap = LinearSegmentedColormap.from_list('colores', ['silver'] + [c[3]['color'] for c in flows], N=len(flows)+1 )

        s = Sankey(flows=flows, cmap=my_cmap, node_pad_y_min=0.15, node_opts=dict(label_format='{value:,.0f}',label_pos='left', label_opts=dict(fontsize=8)))
        for nodes in s.nodes:
            for node in nodes:
                node.label = ''

        fig,ax = plt.subplots(1,1,figsize=(5.5,0.9), layout='constrained')
        s.draw(ax=ax)

        ax.set_title("Number of data in each partition", fontsize=10,y=1.05)  # , x=0.5, y=-0.25
        #ax.text(0.2, 0.98, 'Lateral confinement', fontsize=8, transform=ax.transAxes, ha='center')
        #ax.text(0.5, 0.98, 'Vertical (dis)connection', fontsize=8, transform=ax.transAxes, ha='center')
        #ax.text(0.8, 0.98, 'Sediment regimes', fontsize=8, transform=ax.transAxes, ha='center')
        ax.text(-0.01, 1.3, '(a)', transform=ax.transAxes, ha='left', va='top', fontsize=8)
        #ax.text(-0.7, 0.5, "Partitions of\ndata used\nin analyses", fontsize=8, va='center')

        return fig

    def plot_boxplots_paper(self):

        df = self.src_edz_data.copy()
        edz_fields_map = {
            #'valley_confinement': 'EDZ relative width',
            #'w_edep_scaled': 'Scaled width at the EDZ exit',
            'w_4timesbf_to_w_bf': 'Widths ratio, 4xBF to BF',
            'el_edap_scaled': 'EDZ access stage',
            'wtod_bf': 'Bankfull width-to-depth ratio',
        }
        geo_fields_map = {
            'VC': 'Confinement ratio',
            'IR': 'Incision ratio',
            'WtoD': 'Bankfull width-to-depth ratio'
        #    'ER': 'Entrenchment ratio',
        }
        regs_map1 = {
            'Confined': ['TR','CST'],
            'Unconfined': ['CEFD','DEP','UST','FSTCD'],
        }
        regs_map2 = {
            'Connected': ['CEFD','DEP',],
            'Disconnected': ['UST','FSTCD'],
        }
        regs_map3 = {
            'FSTCD': ['FSTCD',],
            'UST': ['UST',],
        }
        all_colors = self.hexcolors_groups | self.hexcolors_U2021
        porder1 = list(regs_map1.keys())
        colors1 = [all_colors[x] for x in porder1]
        porder2 = list(regs_map2.keys())
        colors2 = [all_colors[x] for x in porder2]
        porder3 = list(regs_map3.keys())
        colors3 = [all_colors[x] for x in porder3]

        edz_features = list(edz_fields_map.keys())
        geo_features = list(geo_fields_map.keys())
        for regk in regs_map1.keys():
            regv = regs_map1[regk]
            mask = df.Ph2SedReg.isin(regv)
            df.loc[mask, 'group1'] = regk

        for regk in regs_map2.keys():
            regv = regs_map2[regk]
            mask = df.Ph2SedReg.isin(regv)
            df.loc[mask, 'group2'] = regk

        for regk in regs_map3.keys():
            regv = regs_map3[regk]
            mask = df.Ph2SedReg.isin(regv)
            df.loc[mask, 'group3'] = regk

        ks_vc = kstest(df[(df.group1 == "Confined")][geo_features[0]].to_numpy(), df[(df.group1 == "Unconfined")][geo_features[0]])[1]
        ks_ir = kstest(df[(df.group2 == "Connected")][geo_features[1]].to_numpy(), df[(df.group2 == "Disconnected")][geo_features[1]])[1]
        ks_wd = kstest(df[(df.group3 == "FSTCD")][geo_features[2]].to_numpy(), df[(df.group3 == "UST")][geo_features[2]])[1]
        ks_edzrw = kstest(df[(df.group1 == "Confined") & ~(df.el_edap_scaled.isnull())][edz_features[0]].to_numpy(), df[(df.group1 == "Unconfined") & ~(df.el_edap_scaled.isnull())][edz_features[0]])[1]
        ks_edzas = kstest(df[(df.group2 == "Connected")][edz_features[1]].to_numpy(), df[(df.group2 == "Disconnected")][edz_features[1]])[1]
        ks_wdbf = kstest(df[(df.group3 == "FSTCD")][edz_features[2]].to_numpy(), df[(df.group3 == "UST")][edz_features[2]])[1]
        print(f"KS test p-value for field {geo_features[0]}: {ks_vc}")
        print(f"KS test p-value for DEM {edz_features[0]}: {ks_edzrw}")
        print(f"KS test p-value for field {geo_features[1]}: {ks_ir}")
        print(f"KS test p-value for DEM {edz_features[1]}: {ks_edzas}")
        print(f"KS test p-value for field {geo_features[2]}: {ks_wd}")
        print(f"KS test p-value for DEM {edz_features[2]}: {ks_wdbf}")


        def field_dem_comparison_plot(df, x, y, hue, ax, colors, legend=False, pval=None, letter=None, log_scale=False,):
            sns.boxplot(df, x=x, y=y, hue=hue, ax=ax, palette=colors, legend=legend, fliersize=0, width=0.5, log_scale=log_scale)
            sns.stripplot(df, x=x, y=y, size=1.5, color='k', alpha=0.5, ax=ax, log_scale=log_scale)

            #sns.violinplot(df, x=x, y=y, hue=hue, ax=ax, palette=colors, legend=legend, log_scale=log_scale, split=True, cut=0, inner='quart', linecolor='k')
            #sns.histplot(df, x=x, y=y, hue=hue, ax=ax, palette=colors, legend=legend, log_scale=log_scale, bins=20)
            #sns.histplot(df, x=x, hue=hue, ax=ax, palette=colors, legend=legend, log_scale=log_scale, multiple='layer', bins=20, binrange=binrange)
            #m1 = df.groupby(y).median(y)[x]
            #def adjust_lightness(color, amount=0.5):
            #    try:
            #        c = mc.cnames[color]
            #    except:
            #        c = color
            #    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            #    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
            #ax.axvline(m1.iloc[0], color=adjust_lightness(colors[0], 0.9), ls='dashed')
            #ax.axvline(m1.iloc[1], color=adjust_lightness(colors[1], 0.9), ls='dashed')

            if pval is not None:
                ax.text(0.975,0.05,f'p={pval:0.2g}', transform=ax.transAxes, ha='right', fontsize=7)
            if letter is not None:
                ax.text(0, 0.975, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontsize=8)
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.grid(True)

        fig,axs = plt.subplots(2,3, figsize=(6,2.5), layout='constrained', gridspec_kw={'wspace': 0.01, 'hspace': 0.01}) 
        axs = axs.ravel()
        lims1 = [0.5,3e2]
        lims2 = [-0.1,4.5]
        lims3 = [8,300]

        edzk = edz_features[0]
        iax = 0
        ax = axs[iax]  # axs[0,1]
        field_dem_comparison_plot(df.sort_values('group1'), x=edzk, y='group1', hue='group1', ax=ax, colors=colors1, pval=ks_edzrw, letter='a', log_scale=True)
        #ax.set_xscale('log')
        ax.set_xlim(lims1)
        ax.set_xlabel(edz_fields_map[edzk])
        ax.set_ylabel('DEM', fontsize=10)
        ax.set_title('Lateral confinement')

        edzk = edz_features[1]
        iax+=1
        ax = axs[iax]  # axs[1,1]
        field_dem_comparison_plot(df.sort_values('group2'), x=edzk, y='group2', hue='group2', ax=ax, colors=colors2, pval=ks_edzas, letter='b')
        ax.set_xlim(lims2)
        ax.set_xlabel(edz_fields_map[edzk])
        ax.set_ylabel('')
        ax.set_title('Floodplain (dis)connection')

        edzk = edz_features[2]
        iax+=1
        ax = axs[iax]  # axs[2,1]
        field_dem_comparison_plot(df.sort_values('group3', ascending=False), x=edzk, y='group3', hue='group3', ax=ax, colors=list(reversed(colors3)), pval=ks_wdbf, letter='c', log_scale=True)
        #ax.set_xscale('log')
        ax.set_xlim(lims3)
        ax.set_xlabel(edz_fields_map[edzk])
        ax.set_ylabel('')
        ax.set_title('FSTCD vs. UST')

        iax +=1
        ax = axs[iax]  # axs[0,0]
        geok = geo_features[0]
        field_dem_comparison_plot(df.sort_values('group1'), x=geok, y='group1', hue='group1', ax=ax, colors=colors1, pval=ks_vc, letter='d', log_scale=True)
        #ax.set_xscale('log')
        ax.set_xlim(lims1)
        ax.set_xlabel(geo_fields_map[geok])
        ax.set_ylabel('Field', fontsize=10)

        iax+=1
        ax = axs[iax]  # axs[1,0]
        geok = geo_features[1]
        field_dem_comparison_plot(df.sort_values('group2'), x=geok, y='group2', hue='group2', ax=ax, colors=colors2, pval=ks_ir, letter='e')
        ax.set_xlim(lims2)
        ax.set_xlabel(geo_fields_map[geok])
        ax.set_ylabel('')

        iax+=1
        ax = axs[iax]  # axs[2,0]
        geok = geo_features[2]
        field_dem_comparison_plot(df.sort_values('group3', ascending=False), x=geok, y='group3', hue='group3', ax=ax, colors=list(reversed(colors3)), pval=ks_wd, letter='f', log_scale=True)
        #ax.set_xscale('log')
        ax.set_xlim(lims3)
        ax.set_yticks([])
        ax.set_xlabel(geo_fields_map[geok])
        ax.set_ylabel('')

        def add_legend(ax, colors, porder):
            lhandles = [mpatches.Patch(color=colors[i], label=porder[i]) for i in range(2)]
            ax.legend(handles=lhandles, loc="upper right", ncol=1, framealpha=0.8, bbox_to_anchor=(1,1), borderpad=0.2, alignment='right', markerfirst=False, fontsize=7)


        for ax,colors,porder in zip(axs[3:], [colors1, colors2, list(reversed(colors3))], [porder1, porder2, list(reversed(porder3))]):
            add_legend(ax, colors, porder)

        fig.supylabel('Data source', fontsize=10)
        #fig.add_artist(mlines.Line2D([0.04, 1], [0.465, 0.465], lw=1, color='lightgrey'))
        fig.add_artist(mlines.Line2D([0.378, 0.378], [0.02, 0.93], lw=1, color='grey'))
        fig.add_artist(mlines.Line2D([0.69, 0.69], [0.02, 0.93], lw=1, color='grey'))

        return fig
    
    def plot_boxplots_supplement(self, edz_features_dict, class1, class2, label1, label2, ax, colors=None, log_scale=False):
        edz_features = list(edz_features_dict.keys())
        chosen_regs = class1 + class2
        mask = self.src_edz_data['Ph2SedReg'].isin(chosen_regs)
        X = self.src_edz_data[edz_features].loc[mask]
        y = self.src_edz_data.Ph2SedReg.loc[mask].isin(class1)
        X['Classes'] = y.map({True: label1, False: label2})
        a = pd.melt(X, id_vars='Classes').sort_values('Classes', key=lambda s: s.apply([label1, label2].index), ignore_index=True)
        #sns.boxplot(a, x='variable', y='value', hue='Classes', ax=ax, palette=colors)
        sns.violinplot(a, x='variable', y='value', hue='Classes', ax=ax, palette=colors, log_scale=log_scale, split=True, cut=0, inner='quart', density_norm='width', linecolor='k')
        ax.get_legend().set_title('')
        labels = []
        for label in ax.get_xticklabels():
            text = edz_features_dict[label.get_text()]
            labels.append(textwrap.fill(text, width=10, break_long_words=False))
        ax.set_xticks(ax.get_xticks(), labels, fontsize=8, rotation=90)
        #ax.set_xticks(ax.get_xticks(), [edz_features_dict[x.get_text()] for x in ax.get_xticklabels()], rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('')
        ax.set_ylabel('Value')

        return ax

    def plot_edz_feature_descriptions(self, reach_id=str(14)):
        reach = self.src_edz_data.loc[reach_id]
        length = reach['length']
        el = self.el_scaled_data[reach_id].to_numpy()
        rh = self.rh_data[reach_id].to_numpy()
        rh_prime = self.rh_prime_data[reach_id].to_numpy()
        rh_prime = gaussian_filter1d(rh_prime.T, 15).T
        width = self.area_data[str(reach_id)].to_numpy() / length
        width = width / 2
        width = np.append(-width[::-1], width)
        section_el = np.append(el[::-1], el)

        el_edap = reach['el_edap_scaled']
        el_edep = reach['el_edep_scaled']

        fig,axs = plt.subplots(3,1, figsize=(6., 8.), layout='constrained')

        ax = axs[0]
        ax.plot(width, section_el, c='k', lw=3, zorder=10000)
        ax.fill_between([width.min(), width.max()], el_edap, el_edep, fc='lightblue', alpha=0.9)
        ax.set_xlim([width.min(), width.max()])
        ax.set_xlabel('Centered width [m]', fontsize=10)

        w4x_1 = np.argmin(np.abs(2*np.abs(width[:width.size//2]) - reach['w_4timesbf']))
        w4x_2 = width.size - w4x_1
        ax.plot([width[w4x_1], width[w4x_2]], [4., 4.,], c='darkorange', lw=2)
        ax.text(-50, 4.1, 'Width at 4x bankfull', c='darkorange', fontsize=10)
        max_exp = np.argmin(rh_prime)
        max_exp_el = el[max_exp]
        wmaxexp_1 = width[width.size//2 - max_exp]
        wmaxexp_2 = width[width.size - (width.size//2 - max_exp)]
        ax.plot([wmaxexp_1, wmaxexp_2], [max_exp_el, max_exp_el], c='darkorange', lw=2)
        ax.text(-50, max_exp_el+0.1, 'Width at max. lateral expansion', c='darkorange', fontsize=10)
        w_bf = reach['w_bf'] / 2
        ax.plot([-w_bf, w_bf], [1, 1], c='darkorange', lw=2)
        ax.text(0, 1.0, 'Bankfull\nwidth', c='darkorange', fontsize=10, ha='center', va='center')




        ax = axs[1]
        ax.plot(rh, el, c='k', lw=3, zorder=10000)
        ax.fill_between([rh.min(), rh.max()], el_edap, el_edep, fc='lightblue', alpha=0.9)
        ax.set_xlim([rh.min(), rh.max()])
        ax.set_xlabel(r'R$_h$ [m]', fontsize=10)

        ax.plot(rh[max_exp], max_exp_el, marker='o', c='purple', zorder=10001)
        rh_edap = np.argmin(np.abs(el - el_edap))
        rh_edep = np.argmin(np.abs(el - el_edep))
        rh_mean = np.mean(rh)
        rh_pre_mean = np.mean(rh[:rh_edap])
        rh_post_mean = np.mean(rh[rh_edep:])
        ax.plot(rh[rh_edap], el[rh_edap], marker='o', c='purple', zorder=10001)
        ax.plot([rh_pre_mean, rh_pre_mean], [0, el[rh_edap]], c='g', ls='dashed', lw=2)
        ax.plot(rh[rh_edep], el[rh_edep], marker='o', c='purple', zorder=10001)
        ax.plot([rh_post_mean, rh_post_mean], [el[rh_edep], 6], c='g', ls='dashed', lw=2)
        ax.plot([rh_mean, rh_mean], [0, 6], c='g', ls='dashed', lw=2)
        ax.text(rh_pre_mean, el_edap, r'Mean R$_h$'+f'\nbelow EDZ', rotation=90, ha='center', va='bottom', c='g', fontsize=10)
        ax.text(rh_mean, (el_edap + el_edep) / 2, r'Mean R$_h$', rotation=90, ha='right', va='center', c='g', fontsize=10)
        ax.text(rh_post_mean, el_edep, r'Mean R$_h$'+f'\nabove EDZ', rotation=90, ha='center', va='top', c='g', fontsize=10)
        ax.text(rh[rh_edap], el_edap, r'R$_h$ at EDZ'+f'\naccess', rotation=90, ha='center', va='bottom', c='purple', fontsize=10)
        ax.text(rh[max_exp], 0.6, r'R$_h$ at max'+f'\nlateral expansion', rotation=90, ha='center', va='center', c='purple', fontsize=10)
        ax.text(rh[rh_edep], el_edep+0.2, r'R$_h$ at'+f'\nEDZ exit', rotation=90, ha='right', va='bottom', c='purple', fontsize=10)



        ax = axs[2]
        ax.plot(rh_prime, el, c='k', lw=3, zorder=10000)
        ax.fill_between([-1,1], el_edap, el_edep, fc='lightblue', alpha=0.9)
        start = np.argmin(np.abs(el - el_edap))
        stop = np.argmin(np.abs(el - el_edep))
        ax.fill_betweenx(el[start:stop], 0.5, rh_prime[start:stop])
        ax.vlines(0.5, el[start], el[stop], ls='solid', color='k', alpha=0.7)
        ax.set_xlim([-1, 1])
        ax.set_xticks([-1.0, -0.5, -0., 0.5, 1.])
        ax.set_xlabel(r"R$_{h}$' [m m$^{-1}$]", fontsize=10)

        rh_prime_max_exp = rh_prime[max_exp]
        rh_prime_mean = np.mean(rh_prime)
        rh_prime_std = np.std(rh_prime)
        #rh_prime_pre_mean = np.mean(rh_prime[:rh_edap])
        #rh_prime_post_mean = np.mean(rh_prime[rh_edep:])

        ax.plot(rh_prime_max_exp, max_exp_el, c='b', marker='o', zorder=10002)
        ax.plot([rh_prime_max_exp, 0.5], [max_exp_el, el_edep], c='r', ls=(0, (1, 1)), lw=2, zorder=10001)
        ax.plot([rh_prime_max_exp, 0.5], [max_exp_el, el_edap], c='r', ls=(0, (1, 1)), lw=2, zorder=10001)
        ax.axvline(rh_prime_mean, ls='dashed', c='green', lw=2)
        ax.annotate('', [rh_prime_mean-rh_prime_std, 4.75], [rh_prime_mean+rh_prime_std, 4.75], arrowprops=dict(arrowstyle='<|-|>', color='darkslategrey', ls='dashed', lw=2))
        #ax.plot([rh_prime_pre_mean, rh_prime_pre_mean], [0, el_edap], ls='dashed', c='green', lw=2)
        #ax.plot([rh_prime_post_mean, rh_prime_post_mean], [el_edep, 6], ls='dashed', c='green', lw=2)
        ax.plot([-1.02, -0.98], [el_edap, el_edap], c='b', lw=2, clip_on=False)
        ax.plot([-1.02, -0.98], [max_exp_el, max_exp_el], c='purple', lw=2, clip_on=False)
        ax.plot([-1.02, -0.98], [el_edep, el_edep], c='b', lw=2, clip_on=False)

        #ax.text(rh_prime_pre_mean-0.03, el_edap, r"Mean R$_h$'"+f'\nbelow EDZ', rotation=90, ha='left', va='bottom', c='g', fontsize=10)
        ax.text(rh_prime_mean+0.02, 2.25, r"Mean R$_h$'", rotation=90, ha='left', va='center', c='g', fontsize=10)
        #ax.text(rh_prime_post_mean-0.02, el_edep+0.4, r"Mean R$_h$'"+f'\nabove EDZ', rotation=90, ha='left', va='top', c='g', fontsize=10)
        ax.text(-0.97, el_edap, 'EDZ access stage', c='b', ha='left', va='center', fontsize=10)
        ax.text(-0.97, max_exp_el, 'Scaled stage of max.\nlateral expansion', c='purple', ha='left', va='center', fontsize=10)
        ax.text(-0.97, el_edep, 'EDZ exit stage', c='b', ha='left', va='center', fontsize=10)
        ax.text((0.5+rh_prime_max_exp)/2, el_edap+0.3, r"R$_h$' slope below"+f'\nmax. lat. exp.', c='r', ha='center', va='top', fontsize=10)
        ax.text((0.5+rh_prime_max_exp)/2, el_edep-0.2, r"R$_h$' slope above"+f'\nmax. lat. exp.', c='r', ha='center', va='bottom', fontsize=10)
        ax.text(rh_prime_max_exp-0.01, max_exp_el, 'Max. lateral\nexpansion', c='b', ha='right', va='center', fontsize=10)
        ax.text(rh_prime_mean-0.02, 4.8, r"$\pm$std. dev."+"\nof Rh'", c='darkslategrey', ha='right', va='center', fontsize=10)
        ax.text(0.5, (el_edap+el_edep)/2, 'Diagnostic size', c='b', ha='right', va='center', fontsize=10)

        letters = ['a', 'b', 'c']
        for ax,l in zip(axs,letters):
            ax.set_ylim([0,6])
            ax.set_ylabel('Scaled stage', fontsize=10)
            ax.text(0.02, 0.9, f'({l})', transform=ax.transAxes, fontsize=10)
        
        #fig.suptitle("DEM-derived features", fontsize=16)
        
        return fig

    def repeat_cv_compare(self, class1, class2, cols_teva, save_fpath, n_iter=10, rng=None, prints=True):

        if rng is None:
            rng = np.random.default_rng()
        idx = self.src_edz_data.Ph2SedReg.isin(class1 + class2)
        y = self.src_edz_data[idx].Ph2SedReg.isin(class1).astype(np.int8)

        try:
            cols_teva.remove('Ph2SedReg')
        except:
            pass
        cols_teva.insert(0, 'Ph2SedReg')
        cols_all = self.edz_data.columns.insert(0, 'Ph2SedReg')
        X_all = self.src_edz_data.loc[idx, cols_all]
        X_teva = self.src_edz_data.loc[idx, cols_teva]

        bas_all = []
        bas_teva = []
        for n in range(n_iter):
            p_idx = rng.permutation(X_all.index)  # X_all and X_teva share the same index
            
            clf_all, cv_res_all = self.rf.rf_classify_with_cv(X_all.iloc[:,1:], y, permute_idx=p_idx, do_fit=False, prints=False)
            m_all = np.mean(cv_res_all['test_score'])
            bas_all.append(m_all)

            clf_teva, cv_res_teva = self.rf.rf_classify_with_cv(X_teva.iloc[:,1:], y, permute_idx=p_idx, do_fit=False, prints=False)
            m_teva = np.mean(cv_res_teva['test_score'])
            bas_teva.append(m_teva)

            if prints:
                print(f'Loop {n+1} of {n_iter}: all columns CV mean {m_all:.3f}, teva columns CV mean {m_teva:.3f}')

        if prints:
            print('Summary using all columns:')
            print(f'Mean of cross-validation means: {np.mean(bas_all)}')
            print(f'Standard deviation: {np.std(bas_all)}')
            print(f'Range: {np.min(bas_all)}, {np.max(bas_all)}')
            print()

            print('Summary using TEVA-selected columns')
            print(f'Mean of cross-validation means: {np.mean(bas_teva)}')
            print(f'Standard deviation: {np.std(bas_teva)}')
            print(f'Range: {np.min(bas_teva)}, {np.max(bas_teva)}')
            print()

        df = pd.DataFrame([bas_all, bas_teva], index=['All features', 'TEVA features']).T
        df.to_csv(save_fpath)

    def repeat_cv_compare_alldata(self, cols_teva, save_fpath, n_iter=10, rng=None, prints=True):

        if rng is None:
            rng = np.random.default_rng()
        
        labs = self.src_edz_data.Ph2SedReg.unique()
        nums = np.arange(len(labs))
        labs_nums_dict = dict(zip(labs, nums))
        y = self.src_edz_data.Ph2SedReg.copy()
        y = y.map(labs_nums_dict).astype(np.int8)

        try:
            cols_teva.remove('Ph2SedReg')
        except:
            pass
        cols_teva.insert(0, 'Ph2SedReg')
        cols_all = self.edz_data.columns.insert(0, 'Ph2SedReg')
        X_all = self.src_edz_data[cols_all]
        X_teva = self.src_edz_data[cols_teva]

        bas_all = []
        bas_teva = []
        for n in range(n_iter):
            p_idx = rng.permutation(X_all.index)  # X_all and X_teva share the same index
            
            clf_all, cv_res_all = self.rf.rf_classify_with_cv(X_all.iloc[:,1:], y, permute_idx=p_idx, do_fit=False, prints=False)
            m_all = np.mean(cv_res_all['test_score'])
            bas_all.append(m_all)

            clf_teva, cv_res_teva = self.rf.rf_classify_with_cv(X_teva.iloc[:,1:], y, permute_idx=p_idx, do_fit=False, prints=False)
            m_teva = np.mean(cv_res_teva['test_score'])
            bas_teva.append(m_teva)

            if prints:
                print(f'Loop {n+1} of {n_iter}: all columns CV mean {m_all:.3f}, teva columns CV mean {m_teva:.3f}')

        if prints:
            print('Summary using all columns:')
            print(f'Mean of cross-validation means: {np.mean(bas_all)}')
            print(f'Standard deviation: {np.std(bas_all)}')
            print(f'Range: {np.min(bas_all)}, {np.max(bas_all)}')
            print()

            print('Summary using TEVA-selected columns')
            print(f'Mean of cross-validation means: {np.mean(bas_teva)}')
            print(f'Standard deviation: {np.std(bas_teva)}')
            print(f'Range: {np.min(bas_teva)}, {np.max(bas_teva)}')
            print()

        df = pd.DataFrame([bas_all, bas_teva], index=['All features', 'TEVA features']).T
        df.to_csv(save_fpath)


class RandomForestCVHandler:

    def __init__(self, rf_kwargs={'n_estimators': 1000, 'oob_score': True}, cv_kwargs={'cv': StratifiedKFold(5), 'return_estimator': True, 'scoring': 'balanced_accuracy'}):
        self.rf_kwargs = rf_kwargs
        self.cv_kwargs = cv_kwargs
    
    def load_clf_cv(self, clf_path, cv_results_path):
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        with open(cv_results_path, 'rb') as f:
            cv_results = pickle.load(f)
        return clf, cv_results

    def permutation_importances(self, clf, X, y, pi_kwargs={'n_repeats': 100, 'n_jobs': 5, 'scoring': 'balanced_accuracy'}, sort_cols=True):

        result = permutation_importance(clf, X, y, **pi_kwargs)
        sorted_importances_idx = np.arange(X.columns.size)
        if sort_cols:
            sorted_importances_idx = result.importances_mean.argsort()

        importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])

        return importances
    
    def rf_classify_with_cv(self, X, y, permute_idx=None, permute_entries=True, rng=None, do_fit=True, prints=True):

        if rng is None:
            rng = np.random.default_rng()

        clf = RandomForestClassifier(**self.rf_kwargs)

        if permute_idx is not None:
            cvX = X.reindex(permute_idx)
            cvy = y.reindex(permute_idx)
        elif permute_entries:
            # shuffle X,y for cross validation
            idx = rng.permutation(X.index)
            cvX = X.reindex(idx)
            cvy = y.reindex(idx)
        else:
            cvX = X
            cvy = y

        cv_results = cross_validate(clf, cvX, cvy.to_numpy().ravel(), **self.cv_kwargs)

        if prints:
            print(f' Mean CV {self.cv_kwargs['scoring']}: {cv_results['test_score'].mean():0.3f}')
            print(f' All CV {self.cv_kwargs['scoring']}: {cv_results['test_score']}')

        # Do the final fit on all of the training data
        if do_fit:
            clf.fit(X,y.to_numpy().ravel())
            if prints:
                try:
                    print(f' Out of bag training score: {clf.oob_score_:0.3f}')
                except:
                    pass

        return clf, cv_results
    
    def save_clf_cv(self, clf, cv_results, out_dir, suffix=''):
        clf_name = 'clf' + str(suffix) + '.pkl'
        cv_name = 'clf_cv' + str(suffix) + '.pkl'
        with open(os.path.join(out_dir, clf_name), 'wb') as f:
            pickle.dump(clf, f)
        with open(os.path.join(out_dir, cv_name), 'wb') as f:
            pickle.dump(cv_results, f)



class TEVAHandler:
    
    def __init__(self, reachdata=None):

        if reachdata is not None:
            self.reachdata = reachdata

    def prep_input_data(self, output_dir, edzs_only=True, binned=False):

        df = self.reachdata.src_edz_data.copy()
        # strip whitespace from all column names
        df.columns = df.columns.str.replace(' ', '')

        # drop unneeded/repeated columns
        drop_cols = ['TotDASqKm', 'Class', 'wbody', 's_order', 'invalid_geometry', 'streamorder','el_bathymetry','el_bathymetry_scaled','rh_bottom','w_bottom',]
        df = df.drop(columns=drop_cols)

        # drop unscaled versions of EDZ variables where scaled version exist
        drop_cols = ['el_edap', 'el_min', 'el_edep', 'height', 'vol', 'w_edap', 'w_edep', ]
        df = df.drop(columns=drop_cols)

        # move classification variables to the front (so we can easily index the predictor variables in columnds )
        df.insert(4, 'Ph2SedReg', df.pop('Ph2SedReg'))
        df.insert(5, 'StreamType', df.pop('StreamType'))
        df.insert(6, 'Bedform', df.pop('Bedform'))
        # but we want slope so move it after them
        df.insert(7, 'slope', df.pop('slope'))

        # For dataset of EDZs ONLY, remove columns  that are from geomorphic data
        if edzs_only:
            df.drop(columns=df.columns[8:51], inplace=True)
            out_suffix = '_edzsonly'
        else:
            out_suffix = ''

        # Top-level: lateral confinement
        # new field: lat_conn = 1, confined (TR, CST); lat_conn = 0, unconfined (CEFD, DEP, FSTCD, UST)
        sub_df0 = df.copy()
        sub_df0['lat_conf'] = 'unconfined'
        sub_df0.loc[df.Ph2SedReg.isin(['TR','CST']), 'lat_conf'] = 'confined'
        sub_df0.to_csv(output_dir + 'Dataset_lat_conf_TEVA' + out_suffix + '.csv', index=True)

        # Mid-level: vertical (dis)connectivity
        sub_df1 = df[df.Ph2SedReg.isin(['CEFD','DEP','FSTCD','UST'])].copy()
        # new field: vert_conn = 1, connected (CEFD, DEP); vert_conn = 0, disconnected (FSTCD, UST)
        sub_df1['vert_conn'] = 'connected'
        sub_df1.loc[sub_df1.Ph2SedReg.isin(['FSTCD','UST']), 'vert_conn'] = 'disconnected'
        sub_df1.to_csv(output_dir + 'Dataset_vert_conn_TEVA' + out_suffix + '.csv', index=True)

        # Finest level: FSTCD and UST stream types
        sub_df2 = df[df.Ph2SedReg.isin(['FSTCD','UST'])].copy()
        # save to CSV
        sub_df2.to_csv(output_dir + 'Dataset_FSTCD_UST_TEVA' + out_suffix + '.csv', index=True)

        # Finest level: TR and CST stream types:
        sub_df3 = df[df.Ph2SedReg.isin(['TR','CST'])].copy()
        # save to CSV
        sub_df3.to_csv(output_dir + 'Dataset_TR_CST_TEVA' + out_suffix + '.csv', index=True)

        # Finest level: CEFD and DEP stream types:
        sub_df4 = df[df.Ph2SedReg.isin(['CEFD','DEP'])].copy()
        # save to CSV
        sub_df4.to_csv(output_dir + 'Dataset_CEFD_DEP_TEVA' + out_suffix + '.csv', index=True)

    def run_teva_model(self, data, classes, output_dir, out_suffix=''):

        # output spreadsheets
        cc_name = output_dir + 'ccs_' + out_suffix + '.xlsx'
        dnf_name = output_dir + 'dnfs_' + out_suffix + '.xlsx'

        # list of input featuers
        input_features_list = data.iloc[:, 7:].columns.tolist()  # data.iloc[:, 7:].columns.tolist()

        # reformat the data
        observation_table = data[input_features_list].to_numpy()

        # Other variables
        n_observations = classes.shape[0]
        n_features = len(input_features_list)
        visualize = False
        output_logging_level = logging.INFO

        cc_max_order = 3  # n_features
        dnf_max_order = 3
        n_cc_gens = 500
        n_dnf_gens = 100
        use_sensitivity = True

        # Algorithm
        teva_alg = teva.TEVA(ccea_max_order                     =cc_max_order, # n_features,
                            ccea_offspring_per_gen             =n_features,
                            ccea_num_new_pop                   =n_features,
                            ccea_total_generations             =n_cc_gens,
                            ccea_n_age_layers                  =5,
                            #  ccea_max_novel_order               =4,
                            ccea_gen_per_growth                =3,
                            ccea_layer_size                    =n_features,
                            ccea_archive_offspring_per_gen     =25,
                            ccea_p_crossover                   =0.5,
                            ccea_p_wildcard                    =0.75,
                            ccea_p_mutation                    =1 / n_features,
                            ccea_tournament_size               =3,
                            ccea_selective_mutation            =False,
                            ccea_use_sensitivity               =use_sensitivity,
                            ccea_sensitivity_threshold         =0,
                            ccea_selection_exponent            =5,
                            ccea_fitness_threshold             =1 / n_observations,
                            ccea_archive_bin_size              =20,

                            dnfea_total_generations            =n_dnf_gens,
                            dnfea_gen_per_growth               =3,
                            dnfea_n_age_layers                 =5,
                            dnfea_offspring_per_gen            =20,
                            dnfea_p_crossover                  =0.5,
                            dnfea_p_targeted_mutation          =0.2,
                            dnfea_p_targeted_crossover         =0.25,
                            dnfea_tournament_size              =3,
                            dnfea_p_union                      =0.5,
                            dnfea_p_intersection               =0.0,
                            dnfea_selection_exponent           =5,
                            dnfea_max_order                    =dnf_max_order,
                            dnfea_layer_size                   =20)
                            # dnfea_max_ccs=4)

        # Run the algorithm for the data set
        unique_classes = teva_alg.fit(observation_table=observation_table,
                                    classifications=classes)

        teva_alg.run_all_targets(logfile_logging_level=logging.INFO,
                                output_logging_level=output_logging_level,
                                visualize=visualize)


        teva_alg.export(cc_name, dnf_name)
        # TEVA export doesn't preserve feature names in the CC output file, for some reason
        # each feature column is labelled like "feature_0"
        # manually resave excel files with the correct feature names
        # they start in column 14

        df_dict = pd.read_excel(cc_name, sheet_name=None)  # sheet_name=None returns a dict with the sheet names as keys and the pandas dataframes as values
        for sheet in df_dict.keys():
            df_dict[sheet].columns = df_dict[sheet].columns[:14].to_list() + input_features_list  # replace column names
            df_dict[sheet].drop(columns=df_dict[sheet].columns[0], inplace=True)

        with pd.ExcelWriter(cc_name) as writer:
            for sheet in df_dict.keys():
                df_dict[sheet].to_excel(writer, sheet_name=sheet)
