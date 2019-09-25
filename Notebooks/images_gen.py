import pandas as pd
import numpy as np
from sompy.sompy import SOMFactory
import sklearn
import sklearn.cluster as cluster
from sklearn import preprocessing
from tfprop_sompy import tfprop_config
from tfprop_sompy import tfprop_vis
from tfprop_sompy import tfprop_analysis
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# this function is used for data processing - creating categories
def classify_by_group(data,prop,k):
    
    """
    This function is to append the normalized PCE_ave data to a 
    new column, prop is the property name string, k is number of 
    group
    """
    
    min_value = min(data[prop])
    max_value = max(data[prop])
    ls = []
    
    for i in range(len(data[prop])):
        norm = (data[prop].iloc[i]-min_value)/(max_value - min_value)
        new_val = np.round(norm * k + 0.499999)
        for n, i in enumerate(ls):
            if i == 0:
                ls[n] = 1
        ls.append(new_val)
    return ls

# or you can use quantile to check the entire dataset based on the quantile of one property
def sort_by_percentile(data,props,quantile):
    """
    Description:
    This function is used to sort data by percentiles
    """
    a = (data[props] > data[props].quantile(quantile))
    
    return pd.DataFrame(data[a])


# This function creates cluster map
def cluster_map(df,labels,sm,n_clusters,savepath):
    """
    Description:
    This function is used to output clustermap
    """
    #n_clusters = 4
    cmap = plt.get_cmap("tab20")
    n_palette = 20  # number of different colors in this color palette
    color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]
    msz = sm.codebook.mapsize
    proj = sm.project_data(sm.data_raw)
    coord = sm.bmu_ind_to_xy(proj)

    fig, ax = plt.subplots(1, 1, figsize=(40,40))

    #cl_labels = som.cluster(n_clusters)
    cl_labels = sklearn.cluster.KMeans(n_clusters = n_clusters, 
                                       random_state = 555).fit_predict(sm.codebook.matrix)

    # fill each rectangular unit area with cluster color
    #  and draw line segment to the border of cluster
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
    ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
                cmap=cmap, norm=norm, edgecolors='face',
                lw=0.5, alpha=0.5)

    ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')
    ax.axis('off')

    for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
        x += 0.2
        y += 0.2
        # "+ 0.1" means shift of label location to upperright direction

        # randomize the location of the label
        #   not to be overwrapped with each other
        # x_text += 0.1 * np.random.randn()
        y += 0.3 * np.random.randn()

        # wrap of label for chemical compound
        #label = str_wrap(label)

        ax.text(x+0.3, y+0.3, label,
                horizontalalignment='left', verticalalignment='bottom',
                rotation=30, fontsize=15, weight='semibold')

    plt.savefig(savepath)

    
# This function prints labels on cluster map
def clusteringmap_category(sm,n_clusters,dataset,colorcategory, savepath):
    """
    Description:
    This function is used to output maps that prints colors on dots based
    on their properties
    """
    categories = dataset[colorcategory] #if colorcategory is one col of the dataset
    cmap = plt.get_cmap("tab20") #cmap for background
    n_palette = 20  # number of different colors in this color palette
    color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]
    msz = sm.codebook.mapsize
    proj = sm.project_data(sm.data_raw)
    coord = sm.bmu_ind_to_xy(proj)

    fig, ax = plt.subplots(1, 1, figsize=(30,30))

    cl_labels = sklearn.cluster.KMeans(n_clusters=n_clusters,random_state=555).fit_predict(sm.codebook.matrix)
        
    # fill each rectangular unit area with cluster color
    #  and draw line segment to the border of cluster
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
#     ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
#                 cmap=cmap, norm=norm, edgecolors='face',
#                 lw=0.5, alpha=0.5)

    ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')
    ax.axis('off')

    categoryname = list(dataset.groupby(colorcategory).count().index)
    categories_int = categories.apply(categoryname.index)

    N = len(categoryname)
    cmap_labels = plt.cm.gist_ncar
    # extract all colors from the .jet map
    cmaplist = [cmap_labels(i) for i in range(cmap_labels.N)]
    # create the new map
    cmap_labels = cmap_labels.from_list('Custom cmap', cmaplist, cmap_labels.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm_labels = mpl.colors.BoundaryNorm(bounds, cmap_labels.N)

    scat = ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c=categories_int,s=300,cmap=cmap_labels,norm=norm_labels)
    cbar = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cbar.ax.get_yaxis().set_ticks([])
    
    for j, lab in enumerate(categoryname):
        cbar.ax.text(1, (2 * j + 1) / (2*(len(categoryname))), lab, ha='left', va='center', fontsize=30)
    cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('# of contacts', rotation=270)
    ax.axis('off')

    for x, y in zip(coord[:, 0], coord[:, 1]):
        x += 0.2
        y += 0.2
        # "+ 0.1" means shift of label location to upperright direction

        # randomize the location of the label
        #   not to be overwrapped with each other
        # x_text += 0.1 * np.random.randn()
        y += 0.3 * np.random.randn()

        # wrap of label for chemical compound
        #label = str_wrap(label)

        # ax.text(x+0.3, y+0.3,horizontalalignment='left', verticalalignment='bottom',rotation=30, fontsize=12, weight='semibold')
    # cl_labels = som.cluster(n_clusters)
    cl_labels = sklearn.cluster.KMeans(n_clusters = n_clusters, 
                                       random_state = 555).fit_predict(sm.codebook.matrix)

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

    plt.savefig(savepath)
    return cl_labels


# This method export images in batches when multiple columns with same starting names exist
def export_img(self,data,begin,som,folder):
    """
    Description:
    This function is to output feature images that starts with 
    same name in batches
    
    data is the data you want to input
    name is the common name the feature starts with
    som is the trained SOM model
    rootpath is the folder you want to save to 
    """
    begin = data[data.columns[pd.Series(data.columns).str.startswith(name)]]
    # end = data[data.columns[pd.Series(data.columns).str.endswith(name)]]
    
    for i in target:
        self.clusteringmap_category(som,list(data.index),7,data,i,folder + i + ".png")


# This function creates image by performing element wise multiplication
def multiply_by(df,sm,n_clusters,path):
    """
    Description:
    This function is to output element 
    wise multiplication of trained SOM varaibles
    """
    codebook = sm._normalizer.denormalize_by(sm.data_raw,sm.codebook.matrix)
    codebookframe = pd.DataFrame(codebook, columns = df.columns)

    ind=-1   # ind will locate the column you just calculated, for example here, E*alpha is the last column 
    codebook = codebookframe.values

    plt.figure(figsize=(10,10))
    ax = plt.subplot(1, 1, 1)
    mp = codebook[:, ind].reshape(sm.codebook.mapsize[0],sm.codebook.mapsize[1]) #
    cmap = plt.get_cmap('RdYlBu_r')

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

    pl = ax.pcolormesh(mp.T, norm=norm, cmap=cmap)
    ax.set_xlim([0, sm.codebook.mapsize[0]])
    ax.set_ylim([0, sm.codebook.mapsize[1]])
    ax.set_aspect('equal')
    ax.set_title("PCE_Ave",fontsize=30)
    ax.tick_params(labelbottom='off')  # turn off tick label
    ax.tick_params(labelleft='off')  # turn off tick label
    ax.tick_params(bottom='off', left='off',
                top='off', right='off')  # turn off ticks

    msz = sm.codebook.mapsize

    #cl_labels = som.cluster(n_clusters)
    cl_labels = sklearn.cluster.KMeans(n_clusters = n_clusters, 
                                       random_state = 555).fit_predict(sm.codebook.matrix)

    for i in range(len(cl_labels)):
        rect_x = [i // msz[1], i // msz[1],
                i // msz[1] + 1, i // msz[1] + 1]
        rect_y = [i % msz[1], i % msz[1] + 1,
                i % msz[1] + 1, i % msz[1]]

        if i % msz[1] + 1 < msz[1]:  # top border
            if cl_labels[i] != cl_labels[i+1]:
                ax.plot([rect_x[1], rect_x[2]],
                        [rect_y[1], rect_y[2]], 'k-', lw=1.5)

        if i + msz[1] < len(cl_labels):  # right border
            if cl_labels[i] != cl_labels[i+msz[1]]:
                ax.plot([rect_x[2], rect_x[3]],
                        [rect_y[2], rect_y[3]], 'k-', lw=1.5)

    plt.colorbar(pl, shrink=0.7)
    plt.show(path)
    plt.savefig(path)

# search aliphatic and aromatic carbons from mols
def search_carbon(data, kind):
    """
    This searches aliphatic and aromatic carbons.
    """
    smis = data.CSMILES 
    count = []
    aliph = Chem.MolFromSmarts("[C]")
    arom = Chem.MolFromSmarts("[c]")
    
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if kind == "aromatic":
            count.append(len(mol.GetSubstructMatches(arom)))
        elif kind == "aliphatic":
            count.append(len(mol.GetSubstructMatches(aliph)))
            
    return count

# sort by range of properties
def sort_by_range(data, min_val, max_val, prop, Categ):
    """
    This function gives values between min and max as 0
    values outside as 1
    """
    ls = []
    df_categ = data[Categ]
    for i in range(len(data[prop])):
        if max_val >= data[prop].iloc[i] >= min_val:
            df_categ.iloc[i] = 0
        else:
            df_categ.iloc[i]  = 1
        ls.append(df_categ.iloc[i])
    return ls
   

# function that tells the length of side chain and whether it is branched
def chain_len_branch(smi):
    """
    :smi is the input SMILES string
    assume there is only one branch 
    """
    import re
    rule = '[0-9HRNGeSiSenrcs@/-=#:[\]{}]' # basic regex rules
    c = re.split(rule, smi)
    bls = [] # branch list
    cls = [] # chain list
    count = 0
    
    for s in c:
        # remove branching symbol '()' from remaining SMILES
        s_pro = s.replace("(","").replace(")","")
        
        if len(s_pro) > 3:
            
            # the remaining are chains with at least 3 atoms, branching or not
            # read content within the parenthesis using string find() method
            
            if "(" and ")" in s:
                
                if s.startswith(")") or s.startswith("]"):
                    s = s[1:]
                
                elif s.endswith("(") or s.endswith("["):
                    s = s[:-1]
                
                # if there are one more multiple parentheses in the segment
                pre_par = s.rfind("(") - s.find("(")
                post_par = s.rfind(")")- s.find(")")
                length = s.rfind(")") - s.find("(") - 3
                
                # (CCCCC)
                if pre_par == 0 and post_par == 0:
                    # Str = re.findall('\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"',s)
                    branch_length = len(s[s.find("(")+1:s.find(")")])
                    chain_length = len(s_pro) - branch_length
                
                # (CCCC)CC)
                elif pre_par == 0 and post_par != 0:
                    loc = s.rfind(")")
                    s = s[:loc] + s[loc+1:]
                    branch_length = len(s[s.find("(")+1:s.find(")")])
                    chain_length = len(s_pro) - branch_length
                
                # (CCCC(CC)
                elif pre_par != 0 and post_par == 0:
                    loc = s.find("(")
                    s = s[:loc] + s[loc+1:]
                    branch_length = len(s[s.find("(")+1:s.find(")")])
                    chain_length = len(s_pro) - branch_length                    
                
                else:
                    # (CCCC)(CCCC)
                    if s.rfind("(") - s.find(")") > 0:
                        # par = re.findall('\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"',s)
                        branch_length = s.find(")") - s.find("(") - 1
                        chain_length = length - branch_length
                    
                    #(CCCC(CC))
                    elif s.rfind("(") - s.find(")") < 0:
                        branch_length = s.find(")") - s.rfind("(") - 1
                        chain_length = length - branch_length
                    
            else:
                chain_length = len(s_pro)
                branch_length = 0
            
            bls.append(branch_length)
            cls.append(chain_length)
    
    return [bls, cls, len(cls)]


# code snippet from the blog "is life worth of living?"
# drawing molecules with index attached to them (Rdkit)
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
 
def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

# mol = Chem.MolFromSmiles( "C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65" )
#Default
#mol
#With index
#mol_with_atom_index(mol)

# from the blog "is life worth of living" 
# high quality image from svg to png
import argparse
import cairosvg
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
 
parser = argparse.ArgumentParser( 'smiles to png inmage' )
parser.add_argument( 'smiles' )
parser.add_argument( '--filename', default="mol." )
DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3.0
 
parser.add_argument( 'smiles' )
 
if __name__=='__main__':
    param = parser.parse_args()
    smiles = param.smiles
    fname = param.filename
    mol = Chem.MolFromSmiles( smiles )
    Draw.MolToFile( mol, fname+"png" )
    Draw.MolToFile( mol, "temp.svg" )
    cairosvg.svg2png( url='./temp.svg', write_to= "svg_"+fname+"png" )