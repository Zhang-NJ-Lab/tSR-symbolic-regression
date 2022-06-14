
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from gplearn import genetic
dataset= r'0107.xlsx'

data=pd.DataFrame(pd.read_excel(dataset))

X = data.values[:,:-1]
y = data.values[:,-1]
for i in range(X.shape[1]):
    X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
'''
est_gp = genetic.SymbolicTransformer(population_size=1000,
                           generations=91, stopping_criteria=0.01,
                           p_crossover=0.8, p_subtree_mutation=0.05,
                           p_hoist_mutation=0.05, p_point_mutation=0.05,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=None,n_components=100)
V=est_gp.fit(X, y)
print(V)
px=V.transform(X)
for i in range(0,50):
  pear=np.corrcoef(px[:,i], y)
  pea=pear[0,1]
  if pea>0.32:
   print(pea,i)'''
   #print(i)


est_gp = genetic.SymbolicTransformer(population_size=1000,
                           generations=91, stopping_criteria=0.01,
                           p_crossover=0.8, p_subtree_mutation=0.05,
                           p_hoist_mutation=0.05, p_point_mutation=0.05,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=None,n_components=100)
V=est_gp.fit(X, y)
#print(V)
px=V.transform(X)
for i in range(0,50):
  pear=np.corrcoef(px[:,i], y)
  pea=pear[0,1]
  if pea>0.32:
   print(pea,i)

for i in range(0,23):
 for j in range(0,23):
  for k in range(0,23):
    for n in range(0,23):
     px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
     per=np.corrcoef(px, y)
     if per[0,1]>0.43 or per[0,1]<-0.43:
      print(per[0,1])
      print(i,j,k,n)
