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
A wrapper for feature selection using any sklearn estimator with feature_importances_ attribute.
"""

#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
import numpy as np

class EmbeddedFilter (BaseEstimator, SelectorMixin):
    """
    A wrapper for feature selection using any sklearn estimator with feature_importances_ attribute.
    Attributes:
        estimator: An sklearn estimator that has a feature_importances_ attribute
        k: Number of features to be selected.
    """
    def __init__(self,  estimator, k):
        self.estimator = estimator
        self.k = k
    def fit(self, X, y=None):
        """
        Apply this embedded feature selection method to (X,y) data.
        :param X: Input data of shape (n_samples, n_features).
        :param y: Target variable of shape (n_samples, ).
        :return: self: object
        """
        self.n_features = np.shape(X)[1]
        if (self.k > self.n_features):
            raise ValueError('No of features to select (%d) is greater than No of features in the data (%d)' %(self.k, self.n_features))

        if (self.k == 0):
            self.k = self.n_features

        self.estimator.fit(X,y)

        if hasattr(self.estimator, "feature_importances_"):
            _scores = self.estimator.feature_importances_
            self.indices = np.argsort(_scores)[::-1]
            self.indices = self.indices[0:self.k]
        return self

    def _get_support_mask(self):
        _mask = np.zeros(self.n_features,dtype=bool)
        for _index in self.indices:
            _mask[_index] = True
        return _mask



