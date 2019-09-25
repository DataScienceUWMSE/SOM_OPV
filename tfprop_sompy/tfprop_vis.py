import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster as clust
import tfprop_sompy
from sompy.visualization.mapview import View2D
from sompy.visualization.umatrix import UMatrixView
from tfprop_sompy import tfprop_config as tfpinit

HUGE = 10000000
labels_rand_seed = 555  # tweak random seed if you do not like labeling result

def kmeans_clust(som, n_clusters=8):
    print("Performing K-means clustering to SOM trained data...")
    cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=tfpinit.km_seed).fit_predict(som.codebook.matrix)

    return cl_labels

def show_posmap(som, placement_name_df, raw_name_df, n_clusters, cl_labels,
                show_data=True, labels=False):
    # user defined color list
    # color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    #               '#ffff33', '#a65628', '#f781bf', '#C71585', '#00FFFF',
    #               '#00FF00', '#F08080', '#DAA520', '#B0E0E6', '#FAEBD7']

    # predefined color map in matplotlib
    # see http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.get_cmap("tab20")
    n_palette = 20  # number of different colors in this color palette
    color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]

    msz = som.codebook.mapsize
    proj = som.project_data(som.data_raw)
    coord = som.bmu_ind_to_xy(proj)

    fig, ax = plt.subplots(1, 1, figsize=tfpinit.posmap_size)

    # fill each rectangular unit area with cluster color
    #  and draw line segment to the border of cluster
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
    ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
                  cmap=cmap, norm=norm, edgecolors='face',
                  lw=0.5, alpha=0.5)
    for i in range(len(cl_labels)):
        rect_x = [i // msz[1], i // msz[1],
                  i // msz[1] + 1, i // msz[1] + 1]
        rect_y = [i % msz[1], i % msz[1] + 1,
                  i % msz[1] + 1, i % msz[1]]

        if i % msz[1] + 1 < msz[1]:  # top border
            if cl_labels[i] != cl_labels[i+1]:
                ax.plot([rect_x[1], rect_x[2]], [rect_y[1], rect_y[2]], 'k-',
                        lw=1.5)

        if i + msz[1] < len(cl_labels):  # right border
            if cl_labels[i] != cl_labels[i+msz[1]]:
                ax.plot([rect_x[2], rect_x[3]], [rect_y[2], rect_y[3]], 'k-',
                        lw=1.5)
    if show_data:
        ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')
        ax.axis('off')

    # place label of each chemical substance
    if labels:
        labels = []
        for i in range(len(raw_name_df)):
            for t in range(len(placement_name_df)):
                if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                    labels.append(raw_name_df.iloc[i, 0])

        # tweak random seed if you do not like labeling result
        np.random.seed(labels_rand_seed)
        for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
            x += 0.1
            y += 0.1
            # "+ 0.1" means shift of label location to upperright direction

            # randomize the location of the label
            #   not to be overwrapped with each other
            # x_text += 0.1 * np.random.randn()
            y += 0.3 * np.random.randn()

            # wrap of label for chemical compound
            label = str_wrap(label)

            ax.text(x+0.5, y+0.5, label,
                    horizontalalignment='left', verticalalignment='bottom',
                    rotation=30, fontsize=14, weight='semibold')

    ax.set_xlim([0, msz[0]])
    ax.set_ylim([0, msz[1]])
    ax.set_aspect('equal')
    ax.tick_params(labelbottom='off')  # turn off tick label
    ax.tick_params(labelleft='off')  # turn off tick label
    ax.grid()
    ax.axis('off')

    plt.show()

    # save figure of SOM positioning map
    if tfpinit.isOutPosmap:
        print("Saving figure of SOM positioning map to {}...".
              format(tfpinit.fout_posmap))
        fig.savefig(tfpinit.fout_posmap)

    return cl_labels


def str_wrap(word):
    import re

    # replace space with linebreak
    word = word.replace(' ', '\n')

    # if find cyclo, fluoro, bromo, fluoro, bromo, or hydro,
    #     linebreak after that
    pattern = r'cyclo|chloro|bromo|fluoro|hydro'
    match = re.search(pattern, word, re.IGNORECASE)
    if match:
        word = word[:match.start()] + word[match.start():match.end()] \
            + '-\n' + word[match.end():]

    # if find methyl with any subsequent letter, linebreak after that
    pattern = r'methyl.'
    match = re.search(pattern, word, re.IGNORECASE)
    if match:
        word = word[:match.start()] + word[match.start():match.end()-1] \
            + '-\n' + word[match.end()-1:]

    return word


class ViewTFP(View2D):

    def _calculate_figure_params(self, som, which_dim, col_sz,
                                 width=None, height=None):
        """ Class method in MapView._calculate_figure_params() overrided """
        codebook = som._normalizer.denormalize_by(som.data_raw,
                                                  som.codebook.matrix)

        indtoshow, sV, sH = None, width, height

        if which_dim == 'all':
            dim = som._dim
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.arange(0, dim).T
            sH, sV = (width, height) or (16, 16*ratio_fig*ratio_hitmap)

        elif type(which_dim) == int:
            dim = 1
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = (width, height) or (16, 16*ratio_hitmap)

        elif type(which_dim) == list:
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = (width, height) or (16, 16*ratio_fig*ratio_hitmap)

        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        self._fig = plt.figure(figsize=(self.width, self.height))
        self._fig.suptitle(self.title)
        plt.rc('font', **{'size': self.text_size})

    def show(self, som, cl_labels, savepath, what='codebook', which_dim='all',
             cmap=None, col_sz=None, desnormalize=False, col_norm=None, 
             isOutHtmap=True):
        """ Class method in View2D.show() overrided """
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = \
            self._calculate_figure_params(som, which_dim, col_sz,
                                          width=self.width, height=self.height)
        self.prepare()
        # Mathtext font to sans-serif
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'sans\-serif'
        mpl.rcParams['mathtext.cal'] = 'sans\-serif'

        cmap = cmap or plt.get_cmap('RdYlBu_r')

        if not desnormalize:
            codebook = som.codebook.matrix
        else:
            codebook = som._normalizer.denormalize_by(som.data_raw,
                                                      som.codebook.matrix)

        if which_dim == 'all':
            names = som._component_names[0]
        elif type(which_dim) == int:
            names = [som._component_names[0][which_dim]]
        elif type(which_dim) == list:
            names = som._component_names[0][which_dim]

        while axis_num < len(indtoshow):
            axis_num += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])

            if col_norm == 'median':  # normalized by median
                min_color_scale = np.median(codebook[:, ind].flatten()) - 1 \
                    * np.std(codebook[:, ind].flatten())
                max_color_scale = np.median(codebook[:, ind].flatten()) + 1 \
                    * np.std(codebook[:, ind].flatten())
                min_color_scale = min_color_scale if min_color_scale >= \
                    np.min(codebook[:, ind].flatten()) \
                    else np.min(codebook[:, ind].flatten())
                max_color_scale = max_color_scale if max_color_scale <= \
                    np.max(codebook[:, ind].flatten()) \
                    else np.max(codebook[:, ind].flatten())
                norm = mpl.colors.Normalize(vmin=min_color_scale,
                                            vmax=max_color_scale,
                                            clip=True)
            else:  # normalized by mean
                min_color_scale = np.mean(codebook[:, ind].flatten()) - 1 \
                    * np.std(codebook[:, ind].flatten())
                max_color_scale = np.mean(codebook[:, ind].flatten()) + 1 \
                    * np.std(codebook[:, ind].flatten())
                min_color_scale = min_color_scale if min_color_scale >= \
                    np.min(codebook[:, ind].flatten()) \
                    else np.min(codebook[:, ind].flatten())
                max_color_scale = max_color_scale if max_color_scale <= \
                    np.max(codebook[:, ind].flatten()) \
                    else np.max(codebook[:, ind].flatten())
                norm = mpl.colors.Normalize(vmin=min_color_scale,
                                            vmax=max_color_scale,
                                            clip=True)

            mp = codebook[:, ind].reshape(som.codebook.mapsize[0],
                                          som.codebook.mapsize[1])
            pl = ax.pcolormesh(mp.T, norm=norm, cmap=cmap)
            ax.set_xlim([0, som.codebook.mapsize[0]])
            ax.set_ylim([0, som.codebook.mapsize[1]])
            ax.set_aspect('equal')
            ax.set_title(names[axis_num - 1])
            ax.tick_params(labelbottom='off')  # turn off tick label
            ax.tick_params(labelleft='off')  # turn off tick label
            ax.tick_params(bottom='off', left='off',
                           top='off', right='off')  # turn off ticks

            plt.colorbar(pl, shrink=0.1)

            # draw line segment to the border of cluster
            msz = som.codebook.mapsize

            for i in range(len(cl_labels)):
                rect_x = [i // msz[1], i // msz[1],
                          i // msz[1] + 1, i // msz[1] + 1]
                rect_y = [i % msz[1], i % msz[1] + 1,
                          i % msz[1] + 1, i % msz[1]]

                if i % msz[1] + 1 < msz[1]:  # top border
                    if cl_labels[i] != cl_labels[i+1]:
                        ax.plot([rect_x[1], rect_x[2]],
                                [rect_y[1], rect_y[2]], 'k-', lw=2.5)

                if i + msz[1] < len(cl_labels):  # right border
                    if cl_labels[i] != cl_labels[i+msz[1]]:
                        ax.plot([rect_x[2], rect_x[3]],
                                [rect_y[2], rect_y[3]], 'k-', lw=2.5)

        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()

        # save figure of heat map
        if isOutHtmap:
            print("Saving figure of heat map for all thermofluid prop. to {}...".format(savepath))
            self._fig.savefig(savepath)


class UMatrixTFP(UMatrixView):

    def show(self, som, placement_name_df, raw_name_df, savepath, 
             distance2=1, row_normalized=False, show_data=True,
             contooor=True, blob=False, labels=False, cmap=None,
             isOutUmat=True):
        """ Class method in UMatrixView.show() overrided """
        umat = self.build_u_matrix(som, distance=distance2,
                                   row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som.project_data(som.data_raw)
        coord = som.bmu_ind_to_xy(proj)

        cmap = cmap or plt.get_cmap('RdYlBu_r')  # set color map

        # colorbar normalization
        min_color_scale = np.mean(umat.flatten()) - 1 * np.std(umat.flatten())
        max_color_scale = np.mean(umat.flatten()) + 1 * np.std(umat.flatten())
        min_color_scale = min_color_scale if min_color_scale >= \
            np.min(umat.flatten()) else np.min(umat.flatten())
        max_color_scale = max_color_scale if max_color_scale <= \
            np.max(umat.flatten()) else np.max(umat.flatten())
        norm = mpl.colors.Normalize(vmin=min_color_scale,
                                    vmax=max_color_scale,
                                    clip=True)

        fig, ax = plt.subplots(1, 1, figsize=(tfpinit.umatrix_size))
        ax.imshow(umat.T, cmap=cmap, alpha=0.7, norm=norm,
                  interpolation='lanczos')

        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            ax.contour(umat.T, np.linspace(mn, mx, 15), linewidths=1.7,
                       cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            ax.scatter(coord[:, 0], coord[:, 1], c='k', marker='o')
            ax.axis('off')

        if labels:
            labels = []
            for i in range(len(raw_name_df)):
                for t in range(len(placement_name_df)):
                    if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                        labels.append(raw_name_df.iloc[i, 0])

            # tweak random seed if you do not like labeling result
            np.random.seed(labels_rand_seed)
            for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
                x += 0.1
                y += 0.1
                # "+ 0.1" means shift of label location to upperright direction

                # randomize the location of the label
                #   not to be overwrapped with each other
                # x_text += 0.1 * np.random.randn()
                y += 0.3 * np.random.randn()

                # wrap of label for chemical compound
                label = str_wrap(label)

                ax.text(x, y, label,
                        horizontalalignment='left', verticalalignment='bottom',
                        rotation=30, fontsize=14, weight='semibold')

        ax.set_xlim([0 - 0.5, msz[0] - 0.5])  # -0.5 for the sake of imshow()
        ax.set_ylim([0 - 0.5, msz[1] - 0.5])
        ax.set_aspect('equal')
        ax.tick_params(labelbottom='off')  # turn off tick label
        ax.tick_params(labelleft='off')  # turn off tick label
        ax.tick_params(bottom='off', left='off',
                       top='off', right='off')  # turn off ticks

        # fig.tight_layout()
        # fig.subplots_adjust(hspace=.0, wspace=.0)
        sel_points = list()

        plt.show()

        # save figure of U-matrix
        if isOutUmat:
            print("Saving figure of U-matrix to {}..."
                  .format(savepath))
            fig.savefig(savepath)

        return sel_points, umat


def potential_func(som, placement_name_df, raw_name_df, gauss_alpha=None,
                   show_data=True, labels=False, cmap=None):
    # predefined color map in matplotlib
    # see http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = cmap or plt.get_cmap("RdYlBu_r")  # colormap for image
    # color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]

    # *** calculate square distance and potential values on each nodes
    nnodes = som.codebook.nnodes
    codebook = som.codebook.matrix
    msz = som.codebook.mapsize
    proj = som.project_data(som.data_raw)
    coord = som.bmu_ind_to_xy(proj)

    # calculate variance of distance between all nodes
    if not gauss_alpha:
        dist_nodes = np.zeros(nnodes * (nnodes-1) // 2)

        k = 0
        for i in range(nnodes - 1):
            for j in range(i+1, nnodes):
                dist_vec = codebook[i] - codebook[j]
                dist_nodes[k] = np.linalg.norm(dist_vec)  # distance
                k += 1
                # print("max. distance: {}, min. distance: {}"
                #       .format(np.max(dist_nodes), np.min(dist_nodes)))
        gauss_alpha = np.std(dist_nodes)
        print("STD of distance between nodes: {}".format(gauss_alpha))

    # calculate potential function
    pot_mean = np.zeros(nnodes)
    npot_sum = np.zeros(nnodes, dtype=int)

    for i in range(nnodes - 1):
        for j in range(i+1, nnodes):
            dist_vec = codebook[i] - codebook[j]
            dist2 = (dist_vec * dist_vec).sum()  # square distance
            gauss_fac = np.exp(-dist2 / (2 * gauss_alpha*gauss_alpha))
            pot_mean[i] += gauss_fac
            pot_mean[j] += gauss_fac
            npot_sum[i] += 1
            npot_sum[j] += 1

    pot_mean = pot_mean / (npot_sum * np.sqrt(2 * np.pi * gauss_alpha))

    # *** plot mean potential values
    fig, ax = plt.subplots(1, 1, figsize=(tfpinit.potfunc_size))
    pl = ax.imshow(pot_mean.reshape(msz[0], msz[1]).T, cmap=cmap,
                   alpha=0.7, interpolation='lanczos')

    fig.colorbar(pl, shrink=0.1)

    if show_data:
        ax.scatter(coord[:, 0], coord[:, 1], c='k', marker='o')
        ax.axis('off')

    if labels:
        labels = []
        for i in range(len(raw_name_df)):
            for t in range(len(placement_name_df)):
                if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                    labels.append(raw_name_df.iloc[i, 0])

        # tweak random seed if you do not like labeling result
        np.random.seed(labels_rand_seed)
        for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
            x += 0.1
            y += 0.1
            # "+ 0.1" means shift of label location to upperright direction

            # randomize the location of the label
            #   not to be overwrapped with each other
            # x_text += 0.1 * np.random.randn()
            y += 0.3 * np.random.randn()

            # wrap of label for chemical compound
            label = str_wrap(label)

            ax.text(x, y, label,
                    horizontalalignment='left', verticalalignment='bottom',
                    rotation=30, fontsize=14, weight='semibold')

    ax.set_xlim([0 - 0.5, msz[0] - 0.5])  # -0.5 for the sake of imshow()
    ax.set_ylim([0 - 0.5, msz[1] - 0.5])
    ax.set_aspect('equal')
    ax.tick_params(labelbottom='off')  # turn off tick label
    ax.tick_params(labelleft='off')  # turn off tick label
    ax.tick_params(bottom='off', left='off',
                   top='off', right='off')  # turn off ticks

    plt.show()

    # save figure of potential surface
    if tfpinit.isOutPot:
        print("Saving figure of potential surface to {}..."
              .format(tfpinit.fout_pot))
        fig.savefig(tfpinit.fout_pot)

    # *** automated clustering based on potential function
    print("Performing potential func. based clustering to SOM trained data...")

    cl_labels = np.empty(nnodes, dtype=int)

    UD2 = som.calculate_map_dist()  # square distance
    distance = 1
    neighborbor_inds = []
    for i in range(nnodes):
        # pick nearest neighborbors
        neighborbor_inds.append([j for j in range(nnodes)
                                 if UD2[i][j] <= distance and j != i])

    n_assigned = 0
    is_assigned = np.zeros(nnodes, dtype=bool)
    is_assigned_tmp = np.zeros(nnodes, dtype=bool)  # temporary for copy
    # masked list of unassigned nodes
    pot_mean_nassign = np.ma.array(pot_mean, mask=is_assigned)
    n_clusters = 0

    while n_assigned < nnodes:
        # search index of max. potential which is not assigned to any cluster
        max_ind = np.ma.argmax(pot_mean_nassign)
        cl_labels[max_ind] = n_clusters
        is_assigned_tmp[max_ind] = True
        n_assigned += 1

        srch_inds = []  # search indices
        srch_inds.append(max_ind)
        # is_valley = np.zeros(nnodes, dtype=bool)  # flag if node is valley
        while len(srch_inds) > 0:
            for i in [j for j in neighborbor_inds[srch_inds[0]]
                      if not is_assigned_tmp[j]]:
                if pot_mean_nassign[srch_inds[0]] >= pot_mean_nassign[i]:
                    cl_labels[i] = n_clusters
                    is_assigned_tmp[i] = True
                    n_assigned += 1

                    srch_inds.append(i)

                else:
                    pass

            # delete searched node
            srch_inds.pop(0)

        n_clusters += 1
        # masked list updated
        is_assigned = np.copy(is_assigned_tmp)
        pot_mean_nassign = np.ma.array(pot_mean, mask=is_assigned)

    # eliminate small cluster
    for i in range(n_clusters):
        clust_list = np.where(cl_labels == i)[0]
        if len(clust_list) <= 3:  # tiny cluster
            n_clusters -= 1
            cl_labels[clust_list] = -1  # -1 assigned to tiny cluster

    print('Number of clusters= {}'.format(n_clusters))
    # print(cl_labels)

    return cl_labels


if __name__ == "__main__":
    import tfprop_som as tfpsom

    codemat_df = pd.read_hdf(tfpsom.fout_train, 'sm_codebook_matrix')
    tfpsom.sm.codebook.matrix = codemat_df.as_matrix()

    # apply matplotlib plotting style
    try:
        plt.style.use(tfpinit.plt_style)
    except OSError:
        print('Warning: cannot find matplotlib style: {}'
              .format(tfpinit.plt_style))
        print('Use default style...')

    # perform K-means clustering
    if tfpinit.isExeKmean:
        cl_labels = kmeans_clust(tfpsom.sm, n_clusters=tfpsom.km_cluster)

        # plot positioning map with clustered groups
        show_posmap(tfpsom.sm, tfpsom.fluid_name_df, tfpsom.fluid_name_df,
                    tfpsom.km_cluster, cl_labels,
                    show_data=True, labels=True)

    # plot potential method and clustering
    if tfpinit.isExePot:
        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        cl_labels = potential_func(tfpsom.sm, tfpsom.fluid_name_df,
                                   tfpsom.fluid_name_df,
                                   gauss_alpha=tfpinit.gauss_alpha,
                                   show_data=True, labels=True,
                                   cmap=cmap)

        # plot positioning map with clustered groups
        show_posmap(tfpsom.sm, tfpsom.fluid_name_df, tfpsom.fluid_name_df,
                    tfpsom.km_cluster, cl_labels,
                    show_data=True, labels=True)

    # plot heat map for each thermofluid property using SOMPY View2D
    if tfpinit.isExeHtmap:
        htmap_x, htmap_y = tfpinit.heatmap_size
        viewTFP = ViewTFP(htmap_x, htmap_y, '', text_size=16)

        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        viewTFP.show(tfpsom.sm, cl_labels, col_sz=tfpinit.heatmap_col_sz,
                     which_dim='all', desnormalize=True, col_norm='median',
                     cmap=cmap)

    # plot U-matrix using SOMPY UMatrixView
    if tfpinit.isExeUmat:
        umatrixTFP = UMatrixTFP(0, 0, '', text_size=14)

        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        umat = umatrixTFP.show(tfpsom.sm, tfpsom.fluid_name_df,
                               tfpsom.fluid_name_df,
                               show_data=True, labels=True, contooor=False,
                               cmap=cmap)
