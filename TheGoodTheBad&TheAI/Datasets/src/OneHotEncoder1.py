#
# OneHotEncoder.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import numpy as np
import pickle


class OneHotEncoder:
    def __init__(self):
        self.tags=[]

    def fit(self,X):
        """Converts list of labels into unique list and stores in self.tags.
        :param X: list of labels
        :return: None
        """
        self.tags   = list(set(X))
        return

    def fit_transform(self,X):
        """Calls fit and transform methods respectively with X.
        :param X: list of labels
        :return: numpy array of one-hot vectors for each element in X
        """
        self.fit(X)
        onehot_vecs = self.transform(X)
        return onehot_vecs

    def transform(self, X):
        """Converts each element in the list into their one-hot representations
        :param X: list of labels
        :return: numpy array of one-hot vectors for each element in X
        """
        size = len(self.tags)
        onehot_vecs = list()
        for x in X:
            onehot_vec          = np.zeros(shape=size)
            if x not in self.tags:
                onehot_vecs.append(onehot_vec)
                continue
            index               = self.tags.index(x)
            onehot_vec[index]   = 1
            onehot_vecs.append(onehot_vec)
        return np.vstack(onehot_vecs)

    def get_feature_names(self):
        """Returns the tags
        :return: tags
        """
        return self.tags

    def decode(self, one_hot_vector):
        """Decodes given one-hot-vector into its value.
        :param one_hot_vector: numpy array for one-hot-vector
        :return: corresponding element in self.tags
        """
        index = one_hot_vector.argmax()
        return self.tags[index]

    def save(self):
        """Saves one-hot encoder to disk"""
        with open('../model/encoder1.p', 'wb') as outfile:
            pickle.dump(self.tags, outfile, pickle.HIGHEST_PROTOCOL)
        print("-> encoder is saved.")
        return

    def load(self):
        """Loads one-hot encoder from disk"""
        with open('Datasets/model/encoder1.p', 'rb') as infile:
            self.tags = pickle.load(infile)
        return
