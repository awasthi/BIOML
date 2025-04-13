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
Helping/Utility functions for RDEA project.
"""

import numpy as np
from sklearn import metrics
from scipy import stats



def auc_relevance(X, y):
    """
    MRMR relevance function using AUC scores.
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :return: AUC score when each variable in X is used to predict y.
    """
    n_samples, n_features = np.shape(X)
    rVal = np.zeros(n_features)
    for i in range(n_features):
        fpr, tpr, thresholds = metrics.roc_curve(y, X[:, i])
        rVal[i] = metrics.auc(fpr, tpr)
    return rVal


def model_based_relevance(model, X, y):
    """
    MRMR relevance function for estimated the relevance of each input variable using feature importance scores of the fitted model.
    :param model:
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :return: Feature importance scores from the model after fitting the input data.
    """
    model = model.fit(X, y)
    if (hasattr(model, 'coef_')):
        rVal = np.fabs(model.coef_)
    elif (hasattr(model, 'feature_importances_')):
        rVal = model.feature_importances_
    else:
        raise ValueError('estimatror does not have coef_  nor feature_importances_ attribute')
    return rVal


def abs_pcc(X, y):
    """
    Returns the absolute
    :param X: Input data of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples, ).
    :return: For each variavle in x, returns the absolute of the Pearson's correlation coeffecient between that variable and target variable.
    """
    _r, _p = stats.pearsonr(X, y)
    return np.abs(_r)


def load_DEGs(dfile):
    """
    Loads a list for differentially expressed genes from a text file.
    :param dfile: Text file including a list of genes (one gene per line).
    :return:
    """
    rVal = []
    with open(dfile) as fp:
        for line in fp:
            rVal.append(line.strip())
    return rVal
