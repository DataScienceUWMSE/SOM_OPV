"""Initial configuration for SOM on thermofluid properties."""

# **** configuration for SOM
# ---- number of thread for training
n_job = 4

# ---- input data file
fin = 'data/fluid_data.csv'

# ---- define list of input thermophysical properties
input_tfprop = ['Name', 'molecular weight', 'melting point', 'boiling point',
                'density', 'viscosity', 'heat capacity', 'vapor pressure',
                'surface tension', 'thermal conductivity']

# ---- alternative column names for thermophysical properties
name_tfprop = None  # do not use alternative name
# name_tfprop = ['molecular weight [kg/kmol]', 'melting point [°C]',
#                'boiling point [°C]', r'density [kg/m$^{3}$]',
#                'viscosity [mPa·s]',
#                'heat capacity [J/(g·K)]', 'vapor pressure [kPa]',
#                'surface tension [mN/m]', 'thermal conductivity [W/(m·K)]']

# ---- size for SOM map
mapsize = (30, 30)

# ---- file name to store trained data (codebook.matrix) as HDF5 format
isOutTrain = True
fout_train = 'data/som_codemat.h5'

# ---- clustering analysis by K-means method
isExeKmean = True
km_cluster = 9  # number of clusters
km_seed = 555  # seed for random number generator for K-mean clustering

# ---- clustering via potential method
isExePot = False
gauss_alpha = None  # gaussian alpha calculated automatically
# gauss_alpha = 1.0  # gaussian alpha parameter

potfunc_size = (10, 10)  # figure size
isOutPot = False
fout_pot = 'data/som_pot.svg'

# **** configuration for visualization
# ---- plottting matplotlib style file
#      (style file should be located on
#       ~/.config/matplotlib/stylelib/{style name}.mplstyle in case of linux)
plt_style = 'myclassic'

# ---- SOM positioning map
posmap_size = (12, 12)  # figure size

isOutPosmap = False
fout_posmap = 'data/som_posmap.svg'

# ---- heat map of all thermofluid properties
heatmap_size = (15, 15)  # figure size

heatmap_col_sz = 4  # number of columns in heat map

isExeHtmap = False
isOutHtmap = True
fout_htmap = 'data/som_htmap.png'

# ---- U-matrix of SOM codebook matrix
umatrix_size = (10, 10)  # figure size

isExeUmat = False
isOutUmat = True
fout_umat = 'data/som_umat.png'

# **** configuration for analysis
# ---- elbow method
isExeElbow = False
isOutElbow = False
fout_elbow = 'data/som_elbow.svg'

# ---- silhouette method
isExeSil = False
sil_clust_range = (5, 10)  # range for number of clusters in silhouette method

isOutSil = False
fout_sil = 'data/som_sil'
fout_silext = 'svg'

# ---- correlation matrix
isExeCorrMat = True
corrmat_size = (12, 9)  # figure size of correlation matrix

isOutCorrMat = False
fout_corrmat = 'data/corr_mat.svg'
