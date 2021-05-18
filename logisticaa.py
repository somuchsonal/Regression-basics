import numpy as np
import pandas as pd
import os,csv,sys,math

df = pd.read_csv(str(sys.argv[1]),header=None)
test = pd.read_csv(str(sys.argv[2]),header=None)
y=pd.get_dummies(df[len(df.columns)-1])
x= pd.get_dummies(df.drop([len(df.columns)-1],1).astype(str))
x.insert(0,'b',1)
xt= pd.get_dummies(test.astype(str))
xt.insert(0,'b',1)
xtest=pd.concat([x,xt] ,sort=False).tail(len(xt))
xtest.fillna(0,inplace=True)
colmat=y.columns
y.columns=range(y.shape[1])
m=y.shape[0]
file1=open(sys.argv[3],"r")
param= file1.readlines()
file1.close()

def cost(y1,yp1):
    return np.mean(-np.sum(np.log(yp1)*y1,axis=1))
w=np.zeros((x.shape[1],y.shape[1]))
costprev=10
costnow=0
i=0
maxi=int(param[2])
etb=0.01
if int(param[0])==1 or int(param[0])==2 :
    eta=-1*float(param[1])
    etb=eta
while i<maxi and abs(costnow-costprev)>1e-5:
    a=np.exp(x.dot(w))
    sums=np.sum(a,axis=1)
    yp=pd.DataFrame()
    for j in range(a.shape[1]):
        yp[j]=a[j]/sums
    costprev=costnow
    costnow=cost(y,yp)
    e=yp-y
    if(int(param[0])==2) : etb=eta/math.sqrt(i+1)
    b=np.multiply(etb/(2*len(x)),x.transpose().dot(e))
    w=b.add(w)
    i+=1
print(i)
at=np.exp(xtest.dot(w))
sumt=np.sum(at,axis=1)
yt=pd.DataFrame()
for j in range(at.shape[1]):
    yt[j]=at[j]/sumt
yt.columns = colmat
ypf=yt.idxmax(axis=1)
np.savetxt(sys.argv[4], ypf, delimiter=",",fmt='%s')
np.savetxt(sys.argv[5], w, delimiter=",")