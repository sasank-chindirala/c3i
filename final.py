import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariateGaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    farray = np.array([])
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs > epsilon)
        f = f1_score(gt, predictions, average = "binary")
        farray = np.append(farray,[f])
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon, epsilons, farray


data=pd.read_csv('SWaT_Dataset_Attack_v0.csv')
data_normal=data[data['Label']==1]
data_anomaly=data[data['Label']==0]
y_normal=data_normal.Label
X_normal=data_normal.drop(['Label',' MV201',' P201',' P202','P203',' P204','P205','P206','MV301','MV302',' MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603',' Timestamp'],axis=1)
y_anomaly=data_anomaly.Label
X_anomaly=data_anomaly.drop(['Label',' MV201',' P201',' P202','P203',' P204','P205','P206','MV301','MV302',' MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603',' Timestamp'],axis=1)

X_train,X_cvandt,y_train,y_cvandt=train_test_split(X_normal,y_normal,test_size=0.4,shuffle=False)
X_cvn,X_testn,y_cvn,y_testn=train_test_split(X_cvandt,y_cvandt,test_size=0.5,shuffle=True)
X_cva,X_testa,y_cva,y_testa=train_test_split(X_anomaly,y_anomaly,test_size=0.5,shuffle=True)


X_cv=pd.concat([X_cvn,X_cva],ignore_index=True)
X_test=pd.concat([X_testn,X_testa],ignore_index=True)
y_cv=pd.concat([y_cvn,y_cva],ignore_index=True)
y_test=pd.concat([y_testn,y_testa],ignore_index=True)

mu, sigma = estimateGaussian(feature_normalize(X_train))
p = multivariateGaussian(feature_normalize(X_train),mu,sigma)
p_cv = multivariateGaussian(feature_normalize(X_cv),mu,sigma)

fscore, ep, epvector, fvector = selectThresholdByCV(p_cv,y_cv)
print(fscore, ep)

p_test = multivariateGaussian(feature_normalize(X_test),mu,sigma)
y_pred= (p_test > ep)
print("F1 score test:",f1_score(y_test,y_pred,average= "binary"))
print("Accuracy score test:",accuracy_score(y_test,y_pred))

plt.figure()
plt.xlabel("Epsilon Value")
plt.ylabel("F1 Score")
plt.plot(epvector,fvector)
plt.show()
