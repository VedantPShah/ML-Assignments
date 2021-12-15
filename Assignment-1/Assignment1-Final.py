
import numpy as np                       # For all our math needs
import matplotlib.pyplot as plt          # For all our plotting needs
# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split


#1
# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y


#2
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise


#3
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')


#4
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')


#Question 1:
    
#1
# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
    Phi=[]
    for value in X:
        Z=[]
        for dim in range(0,d):
            Z.append(np.power(value,dim))
        Phi.append(Z)
    Phi=np.asarray(Phi)
    return Phi

    
#2
# Phi float(n, d): transformed data
# y float(n, ): labels
def train_model(Phi, y):
    w=np.linalg.inv(Phi.T@Phi)@Phi.T@y
    return w


#3
# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    y_predict=Phi@w
    err=(y_predict-y)**2
    Sum=0
    for value in err:
        Sum=Sum+value
    return Sum/len(y)
    

#4
w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models
for d in range(3, 25, 3): # Iterate over polynomial degree
 Phi_trn = polynomial_transform(X_trn, d) # Transform training data into d dimensions
 w[d] = train_model(Phi_trn, y_trn) # Learn model ontraining data

 Phi_val = polynomial_transform(X_val, d) # Transform validation data into d dimensions
 validationErr[d] = evaluate_model(Phi_val, y_val, w[d]) # Evaluate modelon validation data

 Phi_tst = polynomial_transform(X_tst, d) # Transform test data into d dimensions
 testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) # Evaluate model on test data

# Plot all the models
plt.figure()

plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)

plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)

plt.xticks(list(validationErr.keys()), fontsize=12)

plt.legend(['Validation Error', 'Test Error'], fontsize=16)


#5
plt.figure()

plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(3, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(3, 25, 3)))



#Question 2:

    
#1
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    Phi=[]
    for value in X:
        Z=[]
        for dim in range(len(B)):
            Z.append(np.exp(-1*gamma*((value-B[dim])**2)))
        Phi.append(Z)
    Phi=np.asarray(Phi)
    return Phi


#2
# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lmd):
    w=np.linalg.inv(Phi.T@Phi+lmd*np.eye(len(y)))@Phi.T@y
    return w


#3
w1 = {}               # Dictionary to store all the trained models
validationErr1 = {}   # Validation error of the models
testErr1 = {}         # Test error of all the models
lam=[10**n for n in range(-3,4)] 

for d1 in lam:  # Iterate over lamda values from the list
    Phi_trn1 = radial_basis_transform(X_trn,X_trn,gamma=0.1)  # Transform training data                
    w1[d1] = train_ridge_model(Phi_trn1, y_trn,lmd=d1)  # Learn model on training data
        
    Phi_val1 = radial_basis_transform(X_val,X_trn,gamma=0.1) # Transform validation data 
    validationErr1[d1] = evaluate_model(Phi_val1, y_val, w1[d1])  # Evaluate model on validation data
    
    Phi_tst1 = radial_basis_transform(X_tst,X_trn,gamma=0.1) # Transform test data 
    testErr1[d1] = evaluate_model(Phi_tst1, y_tst, w1[d1])  # Evaluate model on test data

# Plot all the models
plt.figure()

plt.plot(validationErr1.keys(), validationErr1.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr1.keys(), testErr1.values(), marker='s', linewidth=3, markersize=12)

plt.xlabel('Lamda', fontsize=12)
plt.ylabel('Validation/Test error', fontsize=12)

plt.xticks(list(validationErr1.keys()),fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=10)
plt.xscale('log')


#4
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d1 in lam:  # Iterate over lamda values from the list
    X_d1 = radial_basis_transform(x_true,X_trn,gamma=0.1)
    y_d1 = X_d1 @ w1[d1]
    plt.plot(x_true, y_d1, marker='None', linewidth=2)
    
plt.legend(['true'] + lam)
