import matplotlib.colors as mc
import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sankeyflow import Sankey


class ReachData:

    def __init__(self, geomorphic_data_path, edz_data_path, drop_unconfined_edz_nan=True):
        df1 = pd.read_csv(geomorphic_data_path)
        df2 = pd.read_csv(edz_data_path)
        df = df1.merge(df2, on='SRC_ID', suffixes=(None, '_y')) # note that columns length, slope, and wbody are duplicated in both dataframes, hence we specify None and _y suffixes
        df = df.drop(columns=df.columns[df.columns.str.contains('_y')])  # the duplicated fields are exactly equal, so we drop the _y columns
        df['SRC_ID'] = df['SRC_ID'].astype(np.int64).astype(str)  # avoids some key issues later
        df.set_index('SRC_ID', inplace=True)

        print(f"Input data shape: {df.shape}")
        if drop_unconfined_edz_nan:
            hand_cliff_reaches = df[(df['el_edap_scaled'].isnull()) & (df['Ph2SedReg'].isin(['CEFD','FSTCD','DEP','UST']))].index
            df = df.drop(index=hand_cliff_reaches)
            print(f"Shape after dropping unconfined NaNs: {df.shape}")
        
        self.src_edz_data = df
    
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

        hexcolors_U2021 = {
            'CEFD': '#019d71b3',
            'CST': '#fce900b3',
            'DEP': '#f798b6b3',
            'FSTCD': '#fe0000b3',
            'TR': '#0b68f0b3',
            'UST': '#ff9901b3',
        }
        regs = [['TR'], ['CEFD'], ['DEP'], ['CST'], ['UST'], ['FSTCD'],]
        titles = [x[0] for x in regs]
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
        for reg,ax,title in zip(regs,axs[1:-1],titles):
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
                color = hexcolors_U2021[tmp_ph2sedreg]
                _ = ax.plot(width, section_el, lw=1, alpha=0.7, color=color,)
            x = np.median(all_widths, axis=0)
            y = np.median(all_els, axis=0)
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
            ax.text(-49.5, 5.9, title, fontsize=10, weight='bold', ha='left', va='top', bbox=dict(boxstyle='square,pad=0', fc='white', ec='none', alpha=0.8), zorder=1001)
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
        cmap_bp = mc.LinearSegmentedColormap.from_list('blue_to_pink', ['#332288', '#ee6576'])
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
        ax.text(0.25, -0.05, "Vertical disconnection", fontsize=8, va="bottom", rotation=90)
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
        labels.append('Median cross section')
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

        plt.show()

        return fig

    def plot_sankey_diagram(self):
        n_conf = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['TR','CST'])].shape[0]
        n_unconf = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CEFD','DEP','FSTCD','UST'])].shape[0]
        n_vconn = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['CEFD','DEP'])].shape[0]
        n_vdisconn = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['FSTCD','UST'])].shape[0]
        n_fstcd = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['FSTCD',])].shape[0]
        n_ust = self.src_edz_data[self.src_edz_data.Ph2SedReg.isin(['UST',])].shape[0]

        flows = [
            ('All data', 'Confined', n_conf, {'color': 'grey'}),
            ('All data', 'Unconfined', n_unconf, {'color': 'lightgrey'}),
            ('Unconfined', 'Vertically\nconnected', n_vconn, {'color': '#332288b3'}),
            ('Unconfined', 'Vertically\ndisconnected', n_vdisconn, {'color': '#ee6576b3'}),
            ('Vertically\ndisconnected', 'UST', n_ust, {'color': '#ff9901b3'}),
            ('Vertically\ndisconnected', 'FSTCD', n_fstcd, {'color': '#fe0000b3'}),
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

        plt.show()

        return fig

