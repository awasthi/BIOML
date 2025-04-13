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
MRMR+ algorithm by Yasser El-Manzalawy.
"""

import numpy as np
import math
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


class MRMRP(BaseEstimator, SelectorMixin):
    """
    For more details about MRMR feature selection algorithm, please see Ding, Chris, and Hanchuan Peng. "Minimum redundancy feature selection from microarray gene expression data." Journal of bioinformatics and computational biology 3.02 (2005): 185-205.
    """

    def __init__(self, relevance_func, redundancy_func, num_features=0, model=None, is_verbose=False):
        """
        Initialize MTMRP object.
        :param relevance_func: relevance function for the MRMR algorithm.
        :param redundancy_func: Redundance function for the MRMR algorithm.
        :param num_features: No of features to be selected.
        :param model: Estimator to be used for model-based relevance function.
        :param is_verbose: if True, more information will be printed.
        """
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.num_features = num_features
        self.model = model
        self.is_verbose = is_verbose

    def fit(self, X, y):
        """
        Run MRMR on (X,y) data.
        :param X:
        :param y:
        :return:
        """
        X = check_array(X, ('csr', 'csc'), dtype=np.float64)

        # TODO: add support for sparse representation

        num_instances, self.num_attributes = np.shape(X)

        if (self.num_features > self.num_attributes):
            raise ValueError('No of features to select (%d) is greater than No of features in the data (%d)' % (
            self.num_features, self.num_attributes))

        if (self.num_features == 0):
            self.num_features = self.num_attributes

        # Calculate self.F

        if (self.model is not None):
            self.F = self.relevance_func(self.model, X, y)
        else:
            self.F = np.zeros(self.num_attributes)
            # calculate rel of each attribute
            try:
                self.F, _ = self.relevance_func(X, y)
            except ValueError:
                self.F = self.relevance_func(X, y)
        self.F /= np.max(self.F)

        if (self.is_verbose):
            print('F computed successfully!')
            print(self.F)

        # start the algorithm

        S = set()
        scores = np.zeros(self.num_attributes)
        ind = np.argmax(self.F)
        S.add(ind)
        scores[ind] = np.amax(self.F)

        self.G = {}
        _tmpr = np.zeros(self.num_attributes)
        for j in range(self.num_attributes):
            try:
                tmp, _ = self.redundancy_func(X[:, ind], X[:, j])
                _tmpr[j] = tmp
            except TypeError:
                tmp = self.redundancy_func(X[:, ind], X[:, j])
                _tmpr[j] = tmp

        self.G[str(ind)] = np.copy(_tmpr)

        for k in range(self.num_features - 1):
            x = np.zeros(self.num_attributes)

            for i in range(self.num_attributes):
                if (i in S):
                    x[i] = float('-inf')
                    continue
                sum = 0.0
                for j in S:
                    sum += self.G[str(j)][i]  # self.G[i, j]
                sum = sum / len(S)  # Note note |S|^2
                x[i] = self.F[i] - sum
            ind = np.argmax(x)
            S.add(ind)
            scores[ind] = (np.amax(x))
            _tmpr = np.zeros(self.num_attributes)
            for j in range(self.num_attributes):
                try:
                    tmp, _ = self.redundancy_func(X[:, ind], X[:, j])
                    _tmpr[j] = tmp
                except TypeError:
                    tmp = self.redundancy_func(X[:, ind], X[:, j])
                    _tmpr[j] = tmp

            self.G[str(ind)] = np.copy(_tmpr)

        min_score = np.amin(scores) - 1
        for i in range(self.num_attributes):
            if (i not in S):
                scores[i] = min_score

        self.feature_importances_ = scores
        self.indices = np.array(list(S))

        return self

    def _get_support_mask(self):
        _mask = np.zeros(self.num_attributes, dtype=bool)
        for _x in self.indices:
            _mask[_x] = True
        return _mask

