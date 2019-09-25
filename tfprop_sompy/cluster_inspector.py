from typing import List
import sompy
from sompy.sompy import SOMFactory

import tfprop_config as tfpinit
import tfprop_vis as tfpvis

import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn.cluster as clust

from IPython.display import display
import ipywidgets as widgets

import pandas as pd
import numpy as np

# FIXME: This is ripped from tfprop_vis. When we re-merge this, this needs to go
def kmeans_clust(som: sompy.sompy.SOM, n_clusters: int=8):
    cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=tfpinit.km_seed).fit_predict(som.codebook.matrix)
    return cl_labels

def sort_materials_by_cluster(mysom: sompy.sompy.SOM, names_df: pd.DataFrame, cl_labels: np.ndarray):
    proj = mysom.project_data(mysom.data_raw)
    coord = mysom.bmu_ind_to_xy(proj)
    
    # Unpack the mapsize coordinate, and reshape `cl_labels` from a flat array to an array shaped like the graph
    # I think the transpose is the right move? Since it'll swap x/y
    msz = mysom.codebook.mapsize
    cl_labels = np.copy(cl_labels).reshape(*msz).T
    
    cluster_lists_dict = {}
    # For all materials, get its coordinate, use that to determine which cluster it is in,
    # And then add it to the appropriate list
    for i, name in enumerate(names_df.index):
        c = coord[i]
        cluster_idx = cl_labels[c[0], c[1]]
        cluster_lists_dict.setdefault(cluster_idx, []).append(name)
        
    # We only know the number of clusters after the fact
    # We build a dictionary first, and when we're done,
    # We turn this dictionary back into a list of lists
    cluster_lists = []
    for i in range(len(cluster_lists_dict.keys())):
        cluster_lists.append(cluster_lists_dict[i])
    return cluster_lists

# cluster number -> color graph
def make_cluster_graph(mysom: sompy.sompy.SOM, cl_labels: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    n_palette = 20
    cmap = plt.get_cmap("tab20")
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
    msz = mysom.codebook.mapsize
    pl = ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
                  cmap=cmap, norm=norm, edgecolors='face',
                  lw=0.5, alpha=0.5)
    plt.colorbar(pl)
    return fig, ax
    
# MIGHT want to add a reference to the graph display instance?
# It'd allow for placing dots on the chart when there's a small amount of selected
# materials remaining, for identifying subgroups/relations
def searchable_dataframe(cluster_df: pd.DataFrame):
    searcher = widgets.Text(description="Search")
    df_output = widgets.Output()
    with df_output:
        print(f"{len(cluster_df)} materials")
        display(cluster_df)

    def handle_searchtext_change(change):
        df_output.clear_output()
        if change.new == "":
            new_display = cluster_df
        else:
            new_display = cluster_df.filter(regex=change.new, axis='index')
        with df_output:
            print(f"{len(new_display)} materials")
            display(new_display)

    searcher.observe(handle_searchtext_change, names='value')
    
    return widgets.VBox([searcher, df_output])
    
# NOTE: Use the following lines before calling this function in order to get a full pandas dataframe display
# (I don't want to leave functions in here that modify global state)
#    pd.set_option('display.max_rows', 2000)
#    pd.set_option('display.width', 1000)
def cluster_tabs(mysom: sompy.sompy.SOM, mats_data_df: pd.DataFrame, clusters_list: List[str]):
    clusters_tabs = widgets.Tab()
    clusters_dataframes = [mats_data_df.filter(l, axis='index') for l in clusters_list]

    graph_output = widgets.Output(layout=widgets.Layout(min_width="40%"))
    # By including the same "graph_output" instance across all tabs,
    # we can render the graph once and have it show up in every tab
    tab_displays = [widgets.HBox([searchable_dataframe(df), graph_output]) 
                    for i, df in enumerate(clusters_dataframes)]
    clusters_tabs.children = tab_displays

    # This is our only way of controlling where to display the graph
    with graph_output:
        display(make_cluster_graph(mysom)[0])

    for i, _ in enumerate(clusters_dataframes):
        clusters_tabs.set_title(i, f"Cluster #{i}")

    return clusters_tabs
    
    
# EXAMPLE OF USING THIS SCRIPT:
# ("sm" is your trained sompy.sompy.SOM instance)
# ("km_clusters" is the number of clusters you want to use)
# ("my_dataframe" is the pandas dataframe that has all your material properties)
# CODE BEGIN:
#  %matplotlib inline
#  # This makes all the loggers stay quiet unless it's important
#  logging.getLogger().setLevel(logging.WARNING)

#  cl_labels = kmeans_clust(sm, km_clusters)
#  clusters_list = sort_materials_by_cluster(sm, cl_labels)

#  # This makes it so it will display the full lists
#  pd.set_option('display.max_rows', 2000)
#  pd.set_option('display.width', 1000)

#  # This should be the last statement of the cell, to make it display
#  # That, or assign the return value to a variable, and have that variable be the final expression in a cell
#  cluster_tabs(sm, my_dataframe, clusters_list)
