import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory

import tfprop_config as tfpinit
import tfprop_vis as tfpvis

# ---- initialization
n_job = tfpinit.n_job
fin = tfpinit.fin
input_tfprop = tfpinit.input_tfprop
mapsize = tfpinit.mapsize
km_cluster = tfpinit.km_cluster
isOutTrain = tfpinit.isOutTrain
fout_train = tfpinit.fout_train

# ---- read data from csv via pandas
fluid_data_df = pd.read_csv(fin,index_col='Name', usecols=input_tfprop)
fluid_data_df = fluid_data_df[input_tfprop[1:]]  # reorder of column
fluid_name_df = pd.DataFrame(fluid_data_df.index)

columns = np.array(tfpinit.name_tfprop) if tfpinit.name_tfprop \
          else np.array(fluid_data_df.columns)
# names = np.array(fluid_name_df)
descr = fluid_data_df.as_matrix()

# make SOM instance
sm = SOMFactory.build(descr, mapsize=mapsize, normalization='var',
                        initialization='pca', component_names=columns)

if __name__ == "__main__":
    # execute SOM training
    sm.train(n_job=n_job, verbose='debug', train_rough_len=0,
             train_finetune_len=0)

    topograpphic_error = sm.calculate_topographic_error()
    quantization_error = np.mean(sm._bmu[1])
    print("Topographic error = {}; Quantization error = {};"
          .format(topograpphic_error, quantization_error))

    # output sm.codebook.matrix as HDF5 format
    if isOutTrain:
        print("Saving SOM trained data to {}...".format(fout_train))
        out_df = pd.DataFrame(sm.codebook.matrix, columns=input_tfprop[1:])
        out_df.to_hdf(fout_train, 'sm_codebook_matrix')

    # apply matplotlib plotting style
    try:
        plt.style.use(tfpinit.plt_style)
    except OSError:
        print('Warning: cannot find matplotlib style: {}'
              .format(tfpinit.plt_style))
        print('Use default style...')

    # perform K-means clustering
    cl_labels = tfpvis.kmeans_clust(sm, n_clusters=km_cluster)

    # plot positioning map with clustered groups
    tfpvis.show_posmap(sm, fluid_name_df, fluid_name_df, km_cluster, cl_labels,
                       show_data=True, labels=True)
