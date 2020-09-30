import re
import numpy as np
import scipy.stats as stats
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.display import display

def clean_s(x):
    """Cleans special characters from a string, preserving case."""
    return re.sub(r'[^\w\s]','',x, re.UNICODE)

def clean_url(u):
    """Regularizes URL strings."""
    u = u.replace('mobile.','').replace('www.','').replace('.com','').replace('.co','').replace('.uk','').\
    replace('.org','').replace('.ca','').replace('youtu.be','youtube').replace('redd.it','reddit')
    u = u.replace('en.','') if u.startswith('en.') else u
    u = u.replace('i.','') if u.startswith('i.') else u
    u = u.replace('m.','') if u.startswith('m.') else u
    return u

def corr_data(df,log_transform=True,do_pairplot=False,pairplot_savename="",disp_subset=None,pairplot_N=None):
    """
    Compute pair-wise correlations among data in `df`.
    
    :param df: dataframe object
    :param log_transform: if True, applies log to all non-negative variables
    :param do_pairplot: if True, generates a pairplot of all variables
    :param pairplot_savename: str filename for pairplot
    :param disp_subset: list containing the subset of variables to correlate (defaults to all variables in `df`) 
    :param pairplot_N: int number of datapoints to sample for plotting the pairplot
    """
    vars_ = df.columns.copy() if not disp_subset else disp_subset
    df_ = df.copy()
    if log_transform:
        var_names = ['log_{}'.format(var) if df[var].min() >= 0 else var for var in vars_]
        for var in var_names:
            if var not in df.columns:
                df_[var] = df[var[4:]].apply(lambda x: np.log(x+0.01))
    else:
        var_names = vars_
        
    if do_pairplot:
        to_plot = df.sample(pairplot_N) if pairplot_N else df
        sns_plot = sns.pairplot(to_plot, 
                                vars=var_names, 
                                diag_kind='kde', height=2.0)
        sns_plot.savefig(pairplot_savename)
        plt.clf() 
        #Image(filename=pairplot_savename)

    rho = df_[var_names].corr().round(2) 
    pval = df_[var_names].corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p_ = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
    display(rho.T.astype(str) + p_.T)
    
    return (rho,pval)
