"""Using scikit learn's GMM classifier"""
import os, sys
import numpy as np
import pylab as pl
from GMM import GMM_Classifier
from sklearn.mixture import gmm
from sklearn import metrics


#----------------------------------------------------------------------
# Load data files
train_data = np.load('data/sdssdr6_colors_class_train.npy')
test_data = np.load('data/sdssdr6_colors_class.200000.npy')

# set the number of training points: using all points leads to a very
# long running time.  We'll start with 10000 training points.  This
# can be increased if desired.
Ntrain = 10000
#Ntrain = len(train_data)

np.random.seed(0)
np.random.shuffle(train_data)
train_data = train_data[:Ntrain]

#----------------------------------------------------------------------
# Split training data into training and cross-validation sets
N_crossval = Ntrain / 5
train_data = train_data[:-N_crossval]
crossval_data = train_data[-N_crossval:]

#----------------------------------------------------------------------
# Set up data
#
X_train = np.zeros((train_data.size, 4), dtype=float)
X_train[:, 0] = train_data['u-g']
X_train[:, 1] = train_data['g-r']
X_train[:, 2] = train_data['r-i']
X_train[:, 3] = train_data['i-z']
y_train = (train_data['redshift'] > 0).astype(int)
Ntrain = len(y_train)

X_crossval = np.zeros((crossval_data.size, 4), dtype=float)
X_crossval[:, 0] = crossval_data['u-g']
X_crossval[:, 1] = crossval_data['g-r']
X_crossval[:, 2] = crossval_data['r-i']
X_crossval[:, 3] = crossval_data['i-z']
y_crossval = (crossval_data['redshift'] > 0).astype(int)
Ncrossval = len(y_crossval)


#   Objects to create:
#    - clf_0 : trained on the portion of the training data with y == 0
#    - clf_1 : trained on the portion of the training data with y == 1


print "Training starts"
clf_0 = gmm.GMM(n_components=2, covariance_type='diag')
clf_1 = gmm.GMM(n_components=2, covariance_type= 'diag')
clf_0.fit([X_train[n] for n in range(len(X_train)) if y_train[n]==0])
clf_1.fit([X_train[n] for n in range(len(X_train)) if y_train[n]==1])
print "Training ends"

# next we must construct the prior.  The prior is the fraction of training
# points of each type.
# 
# variables to compute:
#  - prior0 : fraction of training points with y == 0
#  - prior1 : fraction of training points with y == 1


 
prior0 = float(np.sum(y_train == 0)) / len(y_train)
prior1 = float(np.sum(y_train == 1)) / len(y_train)

# Now we use the prior and the classifiation to compute the log-likelihoods
#  of the cross-validation points.  The log likelihood is given by
#
#    logL(x) = clf.score(x) + log(prior)
#
#  You can use the function np.log() to compute the logarithm of the prior.
#  variables to compute:
#    logL : array, shape = (2, Ncrossval)
#            logL[0] is the log-likelihood for y == 0
#            logL[1] is the log-likelihood for y == 1
logL = None
print "Predicting starts"
logL = np.zeros((2, Ncrossval), dtype=float)
logL[0] = clf_0.score(X_crossval) + np.log(prior0)	 
logL[1] = clf_1.score(X_crossval) + np.log(prior1)



# the predicted value for each sample is the index with the largest
# log-likelihood.
y_pred = np.argmax(logL, 0)
print "Predicting ends"

accuracy = float(np.sum(y_crossval == y_pred)) / len(y_crossval)
print "Accuracy of scikit-learn's GMM classifier: "+ str(accuracy)
