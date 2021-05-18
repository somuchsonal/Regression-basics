import numpy as np
import pandas as pd
import os,csv,sys
from sklearn.linear_model import Lasso

def error(y,yp):
    if y.shape[0] != yp.shape[0]:
        raise Exception('Different number of examples')
    else:
        num=0
        den=0
        for i in range(y.shape[0]):
            num+=(y[i]-yp[i][0])*(y[i]-yp[i][0])
            den+=y[i]*y[i]
        return num/den

df = pd.read_csv(str(sys.argv[2]),header=None)
df[len(df.columns)]=1
test = pd.read_csv(str(sys.argv[3]),header=None)
test[len(test.columns)]=1

if str(sys.argv[1]) == 'a' :
    y=np.array(df[len(df.columns)-2])
    X=np.array(df.drop([len(df.columns)-2],1))
    yt=np.array(test[len(test.columns)-2])
    xt=np.array(test.drop([len(test.columns)-2],1))
    ym=np.reshape(y,(len(df),1))
    m = X.transpose()
    m6= np.linalg.inv(m.dot(X)).dot(m).dot(ym)
    file2 = open(str(sys.argv[5]),"w")
    file1 = open(str(sys.argv[4]),"w")
    yp=xt.dot(m6)
    for i in range(yp.shape[0]):
        if i!=yp.shape[0]-1:
            file1.write(str(yp[i][0])+'\n')
        else : file1.write(str(yp[i][0]))
    file1.close()
    for j in range(m6.shape[0]):
        if j!=m6.shape[0]-1:
            file2.write(str(m6[j][0])+'\n')
        else : file2.write(str(m6[j][0]))
    file2.close()

if str(sys.argv[1]) == 'b' :
    yt=np.array(test[len(test.columns)-2])
    xt=np.array(test.drop([len(test.columns)-2],1))
    file3 = open(str(sys.argv[4]),"r")
    l=file3.readlines()
    emin=0
    lbest=0
    wbest = np.zeros((len(test.columns)-1,1))
    file3.close()
    for n in l:
        try:
            k=float(n)
        except:
            continue
        avge=0
        for i in range(10):
            train = df.drop(np.arange(int(len(df)/10*i),int(len(df)/10*(i+1))),0)
            testb = df[int(len(df)/10*i):int(len(df)/10*(i+1))]
            xtrain=np.array(train.drop([len(train.columns)-2],1))
            ytrain=np.array(train[len(train.columns)-2])
            ymat=np.reshape(ytrain,(len(train),1))
            xtest=np.array(testb.drop([len(testb.columns)-2],1))
            ytest=np.array(testb[len(testb.columns)-2])
            na = xtrain.transpose()
            weight= np.linalg.inv(na.dot(xtrain)+np.multiply(k,np.identity(na.shape[0]))).dot(na).dot(ymat)
            ypred=xtest.dot(weight)
            avge+=error(ytest,ypred)
        if n==l[0] or avge<emin:
            emin=avge
            lbest=k
            wbest=weight
    yp=xt.dot(wbest)
    file5 = open(str(sys.argv[6]),"w")
    file4 = open(str(sys.argv[5]),"w")
    for p in range(yp.shape[0]):
        file4.write(str(yp[p][0])+'\n')
    file4.write(str(lbest))
    file4.close()
    for j in range(wbest.shape[0]):
        if j!=wbest.shape[0]-1:
            file5.write(str(wbest[j][0])+'\n')
        else : file5.write(str(wbest[j][0]))
    file5.close()