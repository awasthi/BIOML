# Copyright (c)  2020, Yasser El-Manzalawy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Yasser El-Manzalawy BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Code for supporting cross-validation experiments
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import interp


def get_labels (probs, cutoff=0.5):
    """
    Binarize predicted probabilities using the specified threhsold
    :param probs: Predicted probabilities.
    :param cutoff: Threshold for converting predicted probabilities into binary predictions.
    :return: Binary labels.
    """
    size  = np.shape(probs)[0]
    labels = np.zeros(size)
    for i in range (size):
        if (probs[i] < cutoff):
            labels[i] = 0
        else:
            labels[i] = 1
    return labels

def evaluate(Y_true, Y_pred, cutoff=0.5):
    """
    Given true and predicted probabilities, return Accuracy, Sensitivity, Specificity,  Matthew Correlation Coefficients, and AUC scores.
    :param Y_true:
    :param Y_pred:
    :param cutoff:
    :return: Accuracy, Sensitivity, Specificity,  Matthew Correlation Coefficients, and AUC scores.
    """
    Y_score = get_labels(Y_pred, cutoff)
    mcc = matthews_corrcoef(Y_true, Y_score)
    acc = accuracy_score(Y_true, Y_score)
    auc = roc_auc_score(Y_true, Y_pred)
    cm = confusion_matrix(Y_true, Y_score, labels=[1,0])
    #print(cm)
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    ap = tp + fn
    an = tn + fp
    total = ap + an
    # compute Sn and Sp
    sn = tp/ap
    sp = tn/an

    # return TP, FN, FP, TN, total, acc, Sn, Sp, MCC, AUC
    return  np.array([acc, sn, sp, mcc, auc])


def do_cross_validation_once(X, y, model, fs_model=None, num_folds=10, random_state = 0,  verbose=False):
    '''
    Run K-fold CV experiment once.
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :param model: sklearn estimator.
    :param fs_model: Feature selection model (optional).
    :param num_folds: No. of folds.
    :param random_state: Random seed.
    :param verbose: if True, more information will be printed.
    :return: Performance metrics averaged over num_folds iterations.
    '''

    # K-fold CV
    skf = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    res = np.zeros(shape=(num_folds, 5))

    base_fpr = np.linspace(0, 1, 201)
    tprs = []

    count = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if (fs_model is not None):
            fs_model = fs_model.fit(X_train, y_train)
            train_filtered = fs_model.transform(X_train)
            test_filtered =  fs_model.transform(X_test)
        else:
            train_filtered = X_train
            test_filtered = X_test

        try:
            Y_pred = model.fit(X=train_filtered, y=y_train).predict_proba(test_filtered)[:, 1]
        except:
            #print('Model has no attribute \'predict_proba\'')
            Y_pred = model.fit(X=train_filtered, y=y_train).predict(test_filtered)
            #print(Y_pred[0:50])

        res[count, :] = evaluate(y_test, Y_pred)
        print(res[count, :])  # Yasser: to track per-fold results

        fpr, tpr, _ = roc_curve(y_test, Y_pred)

        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        count += 1
    if (verbose):
        print(res)
        print(np.mean(res, axis=0))

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    rVal = {}
    rVal['metrics'] = np.mean(res, axis=0)
    rVal['metrics_std'] = np.std(res, axis=0)
    rVal['fpr'] = base_fpr
    rVal['tpr'] = mean_tprs

    return rVal

def do_cross_validation(X, y, model, fs_model=None, num_folds=10, random_state = 0,  n_runs = 1, verbose=False, threshold=0.5):
    """
    Run K-fold CV experiment one or more times.
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :param model: sklearn estimator.
    :param fs_model: Feature selection model (optional).
    :param num_folds: No. of folds.
    :param random_state: Random seed.
    :param n_runs: No. of runs.
    :param verbose: if True, more information will be printed.
    :param threshold: Threshold for transforming predicted probability into a binary label.
    :return: Performance metrics averaged over num_folds*n_runs iterations.
    """

    # K-fold CV
    if (n_runs==1):
        return do_cross_validation_once(X, y, model, fs_model, num_folds, random_state, verbose)

    res = np.zeros(shape=(num_folds * n_runs, 5 )) # 5 is the number of metrics

    base_fpr = np.linspace(0, 1, 51)   #TODO: add a new parameter for 51
    tprs = []


    count = 0
    for cv_run in range(n_runs):
        skf = StratifiedKFold(n_splits=num_folds, random_state=cv_run, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if (fs_model is not None):
                fs_model = fs_model.fit(X_train, y_train)
                train_filtered = fs_model.transform(X_train)
                test_filtered =  fs_model.transform(X_test)
            else:
                train_filtered = X_train
                test_filtered = X_test

            try:
                Y_pred = model.fit(train_filtered, y_train).predict_proba(test_filtered)[:, 1]
            except:
                #print('Model has no attribute \'predict_proba\'')
                Y_pred = model.fit(train_filtered, y_train).predict(test_filtered)


            res[count, :] = evaluate(y_test, Y_pred, cutoff=threshold)




            fpr, tpr, _ = roc_curve(y_test, Y_pred)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

            count += 1
    if (verbose):
        print(res)
        print(np.mean(res, axis=0))

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    rVal = {}
    rVal['metrics'] = np.mean(res, axis=0)
    rVal['metrics_std'] = np.std(res, axis=0)
    rVal['fpr'] = base_fpr
    rVal['tpr'] = mean_tprs

    return rVal


def do_cross_validation_with_feature_selection(X, y, model, fs_model, num_folds=10,  n_runs = 1, verbose=False, threshold=0.5):
    """
    Run K-fold CV experiment one or more times using an estimator and a feature selection method.
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :param model: sklearn estimator.
    :param fs_model: Feature selection model.
    :param num_folds: No. of folds.
    :param random_state: Random seed.
    :param n_runs: No. of runs.
    :param verbose: if True, more information will be printed.
    :param threshold: Threshold for transforming predicted probability into a binary label.
    :return: Performance metrics averaged over num_folds*n_runs iterations and feature importance scores.
    """

  
    #base_fpr = np.linspace(0, 1, 51)   #TODO: add a new parameter for 51
    #tprs = []

    #res = np.zeros(shape=(num_folds * n_runs, 5 ))
    features_importance = np.zeros(np.shape(X)[1])

    res = np.zeros(shape=(num_folds * n_runs, 5 )) # 5 is the number of metrics

    base_fpr = np.linspace(0, 1, 51)   #TODO: add a new parameter for 51
    tprs = []


    count = 0
    for cv_run in range(n_runs):
        skf = StratifiedKFold(n_splits=num_folds, random_state=cv_run, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if (fs_model is not None):
                fs_model = fs_model.fit(X_train, y_train)
                train_filtered = fs_model.transform(X_train)
                test_filtered =  fs_model.transform(X_test)
                for j in fs_model.get_support(indices=True):
                    features_importance[j] += 1
            else:
                raise ValueError('fs_model cannot be None')
            try:
                Y_pred = model.fit(train_filtered, y_train).predict_proba(test_filtered)[:, 1]
            except:
                Y_pred = model.fit(train_filtered, y_train).predict(test_filtered)
            res[count, :] = evaluate(y_test, Y_pred, cutoff=threshold)

            fpr, tpr, _ = roc_curve(y_test, Y_pred)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

            count += 1
    if (verbose):
        print(res)
        print(np.mean(res, axis=0))
    n_iterations = num_folds * n_runs
    features_importance /= n_iterations


    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    rVal = {}
    rVal['metrics'] = np.mean(res, axis=0)
    rVal['metrics_std'] = np.std(res, axis=0)
    rVal['fpr'] = base_fpr
    rVal['tpr'] = mean_tprs
    return rVal, features_importance
