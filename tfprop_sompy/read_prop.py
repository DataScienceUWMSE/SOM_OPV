#read properties and drop useless properties 
import pandas as pd 
import numpy as np 

def read_prop(df,df_prop):
    """
    This function is used to compile a dataframe that has the column as 
    property names and value as property values
    """
    # all cols in read_prop rdkit functions
    vals = (df_prop["prop_value"]) #can't make it to list
    cols = ["exactmw", "lipinskiHBA", "lipinskiHBD", "NumRotatableBonds", "NumHBD", "NumHBA", 
            "NumHeteroatoms", "NumAmideBonds", "FractionCSP3", "NumRings", "NumAromaticRings", "NumAliphaticRings", 
            "NumSaturatedRings", "NumHeterocycles", "NumAromaticHeterocycles", "NumSaturatedHeterocycles", "NumAliphaticHeterocycles", 
            "NumSpiroAtoms", "NumBridgeheadAtoms", "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters", "labuteASA", 
            "tpsa", "CrippenClogP", "CrippenMR"]
    dropcols = ["exactmw","lipinskiHBA","lipinskiHBD","FractionCSP3","labuteASA","tpsa","CrippenClogP","CrippenMR"]
    a = pd.DataFrame(columns=cols)
    for x in vals: #25 names 
        b = pd.DataFrame(data=[x], columns=cols)
        a = a.append(b)
    return pd.DataFrame(a).drop(columns=dropcols).set_index(df["ID No."])