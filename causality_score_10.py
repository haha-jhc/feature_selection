import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
dataframe = pd.read_csv(r'E:\data_all\data_wdz_all_jan_10.csv', engine='python',index_col=0)
data =dataframe.diff().dropna()
X=data.values
a=dataframe.columns.values.tolist()


def var_test(dataframe):
    model = VAR(dataframe)
    # results = model.fit(0)
    results = model.fit(maxlags=14, ic='aic')
    return results


def matrix_cause(results, a):
    mat = np.zeros((13, 13))
    for i in range(0, 13):
        for j in range(0, 13):
            str_x = a[i]
            str_y = a[j]
            b = results.test_causality(str_y, [str_x], kind='f')
            mat[i][j] = 1 - b['pvalue']
    return mat


results=var_test(data)
mat=matrix_cause(results,a)
print(mat[0][0])


#任意的多组列表
da=dict.fromkeys(a)#创建字典存储各维度序列

for i in range(0, 13):
    s = a[i]
    da[s]=mat[i,:]
dataframe_new =pd.DataFrame(da)
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe_new.to_csv("test_10.csv",index=False,sep=',')


