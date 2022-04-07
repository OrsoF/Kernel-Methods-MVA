import pandas as pd
from sift import SIFT
from kernels import KernelSVC_OneVsRest, chi2
from os.path import exists
import zipfile
import numpy as np

if not exists('mva-mash-kernel-methods-2021-2022'):
    with zipfile.ZipFile('mva-mash-kernel-methods-2021-2022.zip', 'r') as zip_ref:
        zip_ref.extractall('mva-mash-kernel-methods-2021-2022')

Xtr = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022\\mva-mash-kernel-methods-2021-2022\\Xtr.csv',header=None,sep=',',usecols=range(3072)))[:3]
Xte = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022\\mva-mash-kernel-methods-2021-2022\\Xte.csv',header=None,sep=',',usecols=range(3072)))[:3]
Ytr = np.array(pd.read_csv('mva-mash-kernel-methods-2021-2022\\mva-mash-kernel-methods-2021-2022\\Ytr.csv',sep=',',usecols=[1])).squeeze()[:3]


sift = SIFT()
X_train_sift, y_train_sift = sift.get_features_from_data(Xtr.reshape((Xtr.shape[0], 32, 32, 3))), Ytr
X_test_sift = sift.get_features_from_data(Xte.reshape((Xte.shape[0], 32, 32, 3)))

n_classes = len(set(Ytr))

clf = KernelSVC_OneVsRest(num_classes = n_classes, C = 0.5, kernel = chi2(gamma = 1.).kernel)
clf.fit(X_train_sift, y_train_sift)

Yte = {'Prediction' : clf.predict(X_test_sift)}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv('Yte.csv',index_label='Id')