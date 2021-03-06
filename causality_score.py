import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


dataframe=pd.read_csv(r'E:\data_all\data_wdz_all_jan_one.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,14,18],engine='python',index_col=0)
X=dataframe.values
a=dataframe.columns.values.tolist()
d=dict.fromkeys(a)#创建字典存储各维度序列
for i in range(0,13):
    s=a[i]
    d[s]=X[:,i]
#VAR_TEST
data =dataframe.diff(1).dropna()
data=data[:1440]
def var_test(dataframe):
    model = VAR(dataframe)
    #results = model.fit(0)
    results = model.fit(maxlags=10, ic='aic')
    return results
 
def matrix_cause(results,a):
    mat=np.zeros((13,13))
    for i in range(0,13):
        for j in range(0,13):
            str_x=a[i]
            str_y=a[j]
            b = results.test_causality(str_y, [str_x], kind='f')
            mat[i][j]=1-b['pvalue']
    return mat
results=var_test(data)
mat=matrix_cause(results,a)
print(mat[1][2])


#任意的多组列表
da=dict.fromkeys(a)#创建字典存储各维度序列

for i in range(0, 13):
    s = a[i]
    da[s]=mat[i,:]
dataframe_new =pd.DataFrame(da)
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe_new.to_csv("test.csv",index=False,sep=',')
