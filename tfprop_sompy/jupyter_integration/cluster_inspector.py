from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import sompy

import sklearn.cluster as clust

from IPython.display import display
import ipywidgets as widgets

from tfprop_sompy.tfprop_vis import render_cluster_borders_to_axes, dataframe_to_coords, render_points_to_axes

import pandas as pd
import numpy as np

class Reference:
    pass

# FIXME: This is ripped from tfprop_vis. When we re-merge this, this needs to go
def kmeans_clust(som: sompy.sompy.SOM, n_clusters: int=8):
    cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=tfpinit.km_seed).fit_predict(som.codebook.matrix)
    return cl_labels

def sort_materials_by_cluster(mysom: sompy.sompy.SOM, names_df: pd.DataFrame, cl_labels: np.ndarray):
    proj = mysom.project_data(mysom.data_raw)
    coord = mysom.bmu_ind_to_xy(proj)
    
    # Unpack the mapsize coordinate, and reshape `cl_labels` from a flat array to an array shaped like the graph
    # Looking at existing datasets, doing the transpose gets the wrong results
    msz = mysom.codebook.mapsize
    cl_labels = np.copy(cl_labels).reshape(*msz)
    
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
    for i, _ in enumerate(cluster_lists_dict.keys()):
        # This is so that the cluster index matches the 
        # order of the entries in the cluster list
        
        # so that you can do `cluster_lists[2]` to get
        # a list of materials in cluster #2
        cluster_lists.append(cluster_lists_dict.get(i, []))
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
    render_cluster_borders_to_axes(ax, cl_labels, msz)
    ax.axis('off')
    fig.colorbar(pl, ax=ax)
    return fig, ax
    
# MIGHT want to add a reference to the graph display instance?
# It'd allow for placing dots on the chart when there's a small amount of selected
# materials remaining, for identifying subgroups/relations
def searchable_dataframe(cluster_df: pd.DataFrame):
    searcher = widgets.Text(description="Search", continuous_update=False)
    df_output = widgets.Output()
    # This is to allow external systems to access the visible dataframe
    # for purposes of performing actions based on its output
    visible_df_reference = Reference()
    visible_df_reference.df = cluster_df
    with df_output:
        print(f"{len(cluster_df)} materials")
        display(cluster_df.describe())
        display(cluster_df)

    def handle_searchtext_change(change):
        df_output.clear_output()
        if change.new == "":
            new_display = cluster_df
        else:
            new_display = cluster_df.filter(regex=change.new, axis='index')
        visible_df_reference.df = new_display
        with df_output:
            print(f"{len(new_display)} materials")
            display(new_display.describe())
            display(new_display)

    searcher.observe(handle_searchtext_change, names='value')
    
    return widgets.VBox([searcher, df_output]), visible_df_reference
    
# NOTE: Use the following lines before calling this function in order to get a full pandas dataframe display
# (I don't want to leave functions in here that modify global state)
#    pd.set_option('display.max_rows', 2000)
#    pd.set_option('display.width', 1000)
def cluster_tabs(mysom: sompy.sompy.SOM, mats_data_df: pd.DataFrame, clusters_list: List[str], cl_labels: np.ndarray):
    clusters_tabs = widgets.Tab()
    clusters_dataframes = [mats_data_df.filter(l, axis='index') for l in clusters_list]

    graph_output = widgets.Output(layout=widgets.Layout(min_width="40%"))
    
    # This uncomfy-looking HTML stuff is to add a color display to let you easily see
    # which color belongs to the cluster you're currently looking at
    def get_tab20_color(index: int):
      tab20_cmap = plt.get_cmap("tab20")
      return tab20_cmap(index/20.)
    def tuple_to_css_rgba(ctup: tuple):
      return f'rgba({255*ctup[0]:.2f}, {255*ctup[1]:.2f}, {255*ctup[2]:.2f}, {ctup[3]:.2f})'

    starter_cluster_fig = make_cluster_graph(mysom, cl_labels)[0]
      
    def _populate_tab(index: int, df: pd.DataFrame):
        df_widget, visible_df = searchable_dataframe(df)

        my_graph_output = widgets.Output(layout=widgets.Layout(min_width="40%"))
        render_points_button = widgets.Button(description="Render points")
        with my_graph_output:
            display(starter_cluster_fig)
        
        def _plot_dataframe_points(mysom: sompy.sompy.SOM, df: pd.DataFrame):
            my_graph_output.clear_output()
            with my_graph_output:
                fig, ax = make_cluster_graph(mysom, cl_labels)
                coords = dataframe_to_coords(mysom, df)
                render_points_to_axes(ax, coords)
                display(fig)
        
        render_points_button.on_click(lambda *args: _plot_dataframe_points(mysom, visible_df.df))
        
        return widgets.HBox([
            widgets.VBox([
              widgets.HBox([
                widgets.Label(value=f"Cluster #{index}"),
                widgets.HTML(
                  layout={'min_width': '20px', 'min_height': '20px'},
                  value=f"<div style='min_width: 20px; min_height: 20px; border: 1px solid black; background-color: {tuple_to_css_rgba(get_tab20_color(index))}; color: {tuple_to_css_rgba(get_tab20_color(index))};'>_</div>"),
              ]),
              render_points_button,
              df_widget]), 
            # By including the same "graph_output" instance across all tabs,
            # we can render the graph once and have it show up in every tab
            my_graph_output])
      
    tab_displays = [ _populate_tab(i, df)
                    for i, df in enumerate(clusters_dataframes)]
    clusters_tabs.children = tab_displays

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
#  cluster_tabs(sm, my_dataframe, clusters_list, cl_labels)
