from sklearn.linear_model import (LinearRegression,Ridge,Lasso,RandomizedLasso)
from sklearn.feature_selection import RFE,f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

dataframe = pd.read_csv(r'E:\data_all\data_wdz-all_jan.csv',usecols=[0,1,2,6,7,14],engine='python',index_col=0)
label=pd.read_csv(r'E:\data_all\data_wdz-all_jan.csv',usecols=[18],engine='python')
X,Y=dataframe.values,label.values

names=["x%s"%i for i in range(1,6)]
ranks={}

def rank_to_dict(ranks,names,order=1):
    minmax=MinMaxScaler()
    ranks=minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks=map(lambda x:round(x,2),ranks)
    return dict(zip(names,ranks))

# lr=LinearRegression(normalize=True)
# lr.fit(X,Y)
# ranks["Linear reg"]=rank_to_dict(np.abs(lr.coef_),names)


# ridge=Ridge(alpha=7)
# ridge.fit(X,Y)
# ranks["ridge"]=rank_to_dict(np.abs(ridge.coef_),names)

lasso=Lasso(alpha=.01)
lasso.fit(X,Y)
ranks["Lasso"]=rank_to_dict(np.abs(lasso.coef_),names)

rlasso = RandomizedLasso(alpha=0.05)
rlasso.fit(X,Y)
ranks["Stability"]=rank_to_dict(np.abs(rlasso.scores_),names)

rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_,names)

r={}
for name in names:
    r[name]=round(np.mean([ranks[method][name] for method in ranks.keys()]),2)
methods = sorted(ranks.keys())
ranks["Mean"]=r
methods.append("Mean")

print ("\t%s" % "\t".join(methods))
for name in names:
    print ("%s\t%s" % (name,"\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))
