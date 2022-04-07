import pandas as pd
from sift import SIFT

Xtr = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022/Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022/Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022/Ytr.csv',sep=',',usecols=[1])).squeeze()

