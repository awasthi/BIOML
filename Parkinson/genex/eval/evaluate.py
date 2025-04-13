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
Code for train and test a model using separate train and test sets.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import interp
from .cross_validation import evaluate


def do_train_test(X_train, y_train, X_test, y_test, model, fs_model=None,  verbose=False, threshold=0.5):
    """
    Train and test an estimator using separate train and test sets.
    :param X_train: Input training data of shape (n_train_samples, n_features).
    :param y_train: Training data target variable of shape (n_train_samples, ).
    :param X_test: Test data of shape (n_samples, n_test_features).
    :param y_test: Test data target variable of shape (n_test_samples, ).
    :param model: sklearn estimator.
    :param fs_model: Feature selection model (optional).
    :param verbose: if True, more information will be printed.
    :param threshold: Threshold for converting predicted probabilities into binary predictions.
    :return: Performance of the model on the test data.
    """

    res = np.zeros(shape=(1, 5))

    base_fpr = np.linspace(0, 1, 201)
    tprs = []

    count = 0

    if (fs_model is not None):
        fs_model = fs_model.fit(X_train, y_train)
        train_filtered = fs_model.transform(X_train)
        test_filtered =  fs_model.transform(X_test)
    else:
        train_filtered = X_train
        test_filtered = X_test

    Y_pred = model.fit(X=train_filtered, y=y_train).predict_proba(test_filtered)[:, 1]
    res[count, :] = evaluate(y_test, Y_pred, cutoff=threshold)

    fpr, tpr, _ = roc_curve(y_test, Y_pred)

    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)


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