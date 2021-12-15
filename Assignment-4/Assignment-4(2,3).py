import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#2
cancer_trn=np.loadtxt(open("/Users/vedantshah/Downloads/wdbc_trn.csv", "rb"), delimiter=",")
X_trn=cancer_trn[:,1:]
y_trn=cancer_trn[:,0]
cancer_tst=np.loadtxt(open("/Users/vedantshah/Downloads/wdbc_tst.csv", "rb"), delimiter=",")
X_tst=cancer_tst[:,1:]
y_tst=cancer_tst[:,0]
cancer_val=np.loadtxt(open("/Users/vedantshah/Downloads/wdbc_val.csv", "rb"), delimiter=",")
X_val=cancer_val[:,1:]
y_val=cancer_val[:,0]

C_range = np.arange(-2.0, 5.0, 1.0)
C_values = np.power(10.0, C_range)

gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()

for C in C_values:
    for G in gamma_values:
        clf = SVC(C=C, kernel='rbf', gamma=G)
        clf.fit(X_trn, y_trn)
        models[C,G] = clf
        trnErr[C,G] = 1-clf.score(X_trn, y_trn)
        valErr[C,G] = 1-clf.score(X_val, y_val)
        tstErr[C,G] = 1-clf.score(X_tst, y_tst)
        
minErr = 1
bestCG = []
for i in valErr:
    if valErr[i] < minErr:
        minErr = valErr[i]
        bestCG = []
        bestCG.append(i)
    elif minErr==valErr[i]:
        bestCG.append(i)

print("List of Best C and Gamma values:",bestCG)

minErr = 1
finalCG = -1
for i in range(len(bestCG)):
    if tstErr[bestCG[i]] < minErr:
        finalCG = bestCG[i]
        minErr = tstErr[bestCG[i]]

print("Best Value for C :",finalCG[0])
print("Best Value for Gamma:", finalCG[1])
print("The accuracy for Best Value for Gamma is:", (1-tstErr[finalCG[0],finalCG[1]])*100,"%")


#3
models1 = dict()
trnErr1 = dict()
valErr1 = dict()
tstErr1 = dict()

k_values = [1,5,11,15,21]

for K in k_values:
    kd = KNeighborsClassifier(n_neighbors=K,algorithm='kd_tree')
    kd.fit(X_trn,y_trn)
    models1[K] = kd
    trnErr1[K] = 1-kd.score(X_trn, y_trn)
    valErr1[K] = 1-kd.score(X_val, y_val)
    tstErr1[K] = 1-kd.score(X_tst, y_tst)
    
plt.figure()
plt.plot(valErr1.keys(), valErr1.values(), marker='o', linewidth=3, markersize=12)
plt.plot(trnErr1.keys(), trnErr1.values(), marker='s', linewidth=3, markersize =12)
plt.xlabel('K value', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(valErr1.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.xscale('log')

minErr1 = 1
bestK = []
for i in valErr1:
    if valErr1[i] < minErr1:
        minErr1 = valErr1[i]
        bestK = []
        bestK.append(i)
    elif minErr1==valErr1[i]:
        bestK.append(i)

print("List of Best K values:",bestK)

        
minErr1 = 1
finalK = -1
for i in range(len(bestK)):
    if tstErr1[bestK[i]] < minErr1:
        finalK = bestK[i]
        minErr1 = tstErr1[bestK[i]]

print("Best K value is:", finalK)
print("The accuracy for Best K value is ",(1-tstErr1[finalK])*100, "%")
