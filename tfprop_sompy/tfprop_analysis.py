import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from tfprop_sompy import tfprop_config as tfpinit

def elbow(som, max_cluster=10):
    from sklearn.cluster import KMeans
    distortions = []
    for i in range(1, max_cluster+1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,
                    random_state=tfpinit.km_seed).fit(som.codebook.matrix)
        distortions.append(km.inertia_)
        print("Number of clusters: {:3d}, SSE= {:.2f}".format(i, km.inertia_))

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(1, max_cluster+1), distortions, 'o-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('SSE in clusters')
    ax.set_xlim([1, max_cluster])

    # use only integer value on x axis
    # ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # exponent expression for y-axis tick label
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'sans\-serif'
    mpl.rcParams['mathtext.cal'] = 'sans\-serif'
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.show()

    if tfpinit.isOutElbow:
        print("Saving figure of elbow method to {}...".
              format(tfpinit.fout_elbow))
        fig.savefig(tfpinit.fout_elbow)


def silhouette(som, n_cluster=4, cmap=None):
    from sklearn.metrics import silhouette_samples
    from sklearn.cluster import KMeans

    y_km = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10,
                  max_iter=300,
                  random_state=tfpinit.km_seed).fit_predict(som.codebook.matrix)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]

    silhouette_vals = silhouette_samples(som.codebook.matrix, y_km,
                                         metric='euclidean')

    # set colormap from predefined colormap
    cmap = cmap or plt.get_cmap('viridis')

    # create figure instance
    fig, ax = plt.subplots(1, 1)

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cmap(float(i) / n_clusters)
        ax.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals,
                height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    print("Number of clusters: {:3d}, silhouette coeff.: {:5.2f}"
          .format(n_clusters, silhouette_avg))
    ax.axvline(silhouette_avg, color='red', linestyle='--')
    ax.set_xlim(right=1.0)
    ax.set_ylim(0)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i+1) for i in cluster_labels])
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Silhoutte coefficient')

    plt.show()

    if tfpinit.isOutSil:
        # set file name of figure
        fout_sil = "{}{}.{}".format(tfpinit.fout_sil, str(n_clusters),
                                    tfpinit.fout_silext)
        print("Saving figure of silhouette method to {}...".
              format(fout_sil))
        fig.savefig(fout_sil)

    plt.close(fig)


def silhouette_range(som, sil_range=(1, 1), cmap=None):
    sil_start, sil_end = sil_range
    sil_end += 1

    for i in range(sil_start, sil_end):
        silhouette(som, i, cmap)


def corr_matrix(prop_df, usecols=[], cmap=None):

    # calculate correlation matrix from pandas DaraFrame
    corr_mat = np.corrcoef(prop_df[usecols].values.T)
    # The following one also works well and return DataFrame (might be slower)
    # corr_mat = prop_df[usecols].corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # create figure instance
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=tfpinit.corrmat_size)

    # Generate a Seaborn colormap
    cmap = cmap or sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    cax = fig.add_axes([0.75, 0.2, 0.02, 0.5])  # shift color bar
    sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.3,
                square=True, linewidths=.5,
                cbar_ax=cax,
                # cbar_kws={'shrink': .6},
                # xticklabels=usecols, yticklabels=usecols,
                annot=True, annot_kws={'fontsize': 10}, fmt='.2f', ax=ax)
    # empty string to last element of usecols
    xlabels = np.append(usecols[:-1], '')
    ax.set_xticklabels(xlabels, rotation=45)
    # empty string to first element of usecols and inverse order of list
    ylabels = np.append(usecols[:0:-1], '')
    ax.set_yticklabels(ylabels, rotation=45)
    ax.tick_params(bottom='off', left='off',
                   top='off', right='off')  # turn off ticks

    plt.show()

    if tfpinit.isOutCorrMat:
        print("Saving figure of correlation matrix to {}...".
              format(tfpinit.fout_corrmat))
        fig.savefig(tfpinit.fout_corrmat)


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

    # Analysis on number of clusters by elbow method
    if tfpinit.isExeElbow:
        print('*** Performing clustering analysis of elbow method...')
        elbow(tfpsom.sm, max_cluster=20)

    # Analysis on number of clusters by silhouette method
    if tfpinit.isExeSil:
        print('*** Performing clustering analysis of silhouette method...')

        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        silhouette_range(tfpsom.sm, tfpinit.sil_clust_range, cmap)

    # Analysis on correlation matrix by Seaborn
    if tfpinit.isExeCorrMat:
        print('*** Generating correlation matrix plotted by Seaborn...')

        cmap = 'RdBu_r'  # set color map name
        corr_matrix(tfpsom.fluid_data_df, usecols=tfpsom.columns, cmap=cmap)
