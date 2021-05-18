import numpy as np
import pandas as pd
import os,csv,sys,math

df = pd.read_csv(str(sys.argv[1]),header=None)
test = pd.read_csv(str(sys.argv[2]),header=None)
y=pd.get_dummies(df[len(df.columns)-1])
x= pd.get_dummies(df.drop([len(df.columns)-1],1).astype(str))
x.insert(0,'b',1)
xnames=['b','0_usual','0_pretentious','0_great_pret','1_proper', '1_less_proper','1_improper', '1_critical',
        '1_very_crit','2_complete', '2_completed', '2_incomplete', '2_foster','3_1', '3_2', '3_3','3_more',
        '4_convenient','4_less_conv','4_critical', '5_convenient', '5_inconv','6_nonprob','6_slightly_prob',
        '6_problematic','7_recommended','7_priority','7_not_recom' ]
x = x[xnames]
xt= pd.get_dummies(test.astype(str))
xt.insert(0,'b',1)
xtest=pd.concat([x,xt] ,sort=False).tail(len(xt))
xtest.fillna(0,inplace=True)
ynames = ['not_recom','recommend','very_recom','priority','spec_prior']
y=y[ynames]
y.columns=range(y.shape[1])
m=y.shape[0]
file1=open(sys.argv[3],"r")
param= file1.readlines()
file1.close()

def cost(y1,yp1):
    return np.mean(-np.sum(np.log(yp1)*y1,axis=1))/10

w=np.zeros((x.shape[1],y.shape[1]))
costprev=10
costnow=0
maxi=int(param[2])
etb=1
if int(param[0])==1 or int(param[0])==2 :
    eta=float(param[1])
    etb=eta
if int(param[0])==3 :
    alpha = float(param[1].split(',')[0])
    beta = float(param[1].split(',')[1])
for i in range(maxi) :
    a=np.exp(x.dot(w))
    sums=np.sum(a,axis=1)
    yp=a.div(sums,axis=0)
    if int(param[0])==2 : etb=eta/math.sqrt(i+1)
    such = etb*x.transpose().dot(y-yp)
    b=such/len(x)
    if int(param[0])==3 :
        costprev=costnow
        costnow=cost(y,yp)
        st1=np.array(such).reshape((such.shape[0]*such.shape[1],1))
        c=-1*alpha*etb*(st1.transpose().dot(st1))
        if costnow>costprev+c :
           etb = etb*beta
    w=w+b
print(i+1)
at=np.exp(xtest.dot(w))
sumt=np.sum(at,axis=1)
yt=at.div(sumt,axis=0)
yt.columns = ynames
ypf=yt.idxmax(axis=1)
np.savetxt(sys.argv[4], ypf, delimiter=",",fmt='%s')
np.savetxt(sys.argv[5], w, delimiter=",")