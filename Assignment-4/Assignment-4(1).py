# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION, 
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
  # Generate a non-linear data set
  X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
   
  # Take a small subset of the data and make it VERY noisy; that is, generate outliers
  m = 30
  np.random.seed(30)  # Deliberately use a different seed
  ind = np.random.permutation(n_samples)[:m]
  X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
  y[ind] = 1 - y[ind]

  # Plot this data
  cmap = ListedColormap(['#b30065', '#178000'])  
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
  
  # First, we use train_test_split to partition (X, y) into training and test sets
  X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, 
                                                random_state=42)

  # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
  X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, 
                                                random_state=42)
  
  return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION, 
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT 
#

def visualize(models, param, X, y):
  # Initialize plotting
  if len(models) % 3 == 0:
    nrows = len(models) // 3
  else:
    nrows = len(models) // 3 + 1
    
  fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
  cmap = ListedColormap(['#b30065', '#178000'])

  # Create a mesh
  xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
  yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), 
                             np.arange(yMin, yMax, 0.01))

  for i, (p, clf) in enumerate(models.items()):
    # if i > 0:
    #   break
    r, c = np.divmod(i, 3)
    ax = axes[r, c]

    # Plot contours
    zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

    if (param == 'C' and p > 0.0) or (param == 'gamma'):
      ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], 
                 alpha=0.5, linestyles=['--', '-', '--'])

    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
    ax.set_title('{0} = {1}'.format(param, p))

    
# Generate the data
n_samples = 500    # Total size of data set 
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)


#a
# Learn support vector classifiers with a radial-basis function kernel with 
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()

for C in C_values:
  clf = SVC(C=C, kernel='rbf', gamma='scale')
  clf.fit(X_trn, y_trn)
  models[C] = clf
  trnErr[C] = 1-clf.score(X_trn, y_trn)
  valErr[C] = 1-clf.score(X_val, y_val)
  tstErr[C] = 1-clf.score(X_tst, y_tst)
   
  
visualize(models, 'C', X_trn, y_trn)

plt.figure()
plt.plot(valErr.keys(), valErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(trnErr.keys(), trnErr.values(), marker='s', linewidth=3, markersize =12)
plt.xlabel('C value', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Train Error'], fontsize=16)
plt.xscale('log')


minErr = 1
bestC = []
for i in valErr:
    if valErr[i] < minErr:
        minErr = valErr[i]
        bestC = []
        bestC.append(i)
    elif minErr==valErr[i]:
        bestC.append(i)

print("List of Best C values on Validation:",bestC)

minErr = 1
finalC = -1
for i in range(len(bestC)):
    if tstErr[bestC[i]] < minErr:
        finalC = bestC[i]
        minErr = tstErr[bestC[i]]

print("Best C value is:", finalC)
print("The accuracy for est C value is:", (1-tstErr[finalC])*100,"%")


#b
# Learn support vector classifiers with a radial-basis function kernel with 
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr1 = dict()
valErr1 = dict()
tstErr1 = dict()

for G in gamma_values:
    clf = SVC(C=10, kernel='rbf', gamma=G)
    clf.fit(X_trn, y_trn)
    models[G] = clf
    trnErr1[G] = 1-clf.score(X_trn, y_trn)
    valErr1[G] = 1-clf.score(X_val, y_val)
    tstErr1[G] = 1-clf.score(X_tst, y_tst)
  
visualize(models, 'gamma', X_trn, y_trn)

plt.figure()
plt.plot(valErr1.keys(), valErr1.values(), marker='o', linewidth=3, markersize=12)
plt.plot(trnErr1.keys(), trnErr1.values(), marker='s', linewidth=3, markersize =12)
plt.xlabel('Gamma value', fontsize=16)
plt.ylabel('Validation/Train error', fontsize=16)
plt.xticks(list(valErr1.keys()), fontsize=12)
plt.legend(['Validation Error', 'Train Error'], fontsize=16)
plt.xscale('log')


minErr1 = 1
bestG = []
for i in valErr1:
    if valErr1[i] < minErr1:
        minErr1 = valErr1[i]
        bestG = []
        bestG.append(i)
    elif minErr1==valErr1[i]:
        bestG.append(i)

print("List of Best Gamma values on Validation:",bestG)

minErr1 = 1
finalG = -1
for i in range(len(bestG)):
    if tstErr1[bestG[i]] < minErr1:
        finalG = bestG[i]
        minErr1 = tstErr1[bestG[i]]

print("Best Gamma value is:", finalG)
print("The accuracy for Best Gamma value is:", (1-tstErr1[finalG])*100,"%")
