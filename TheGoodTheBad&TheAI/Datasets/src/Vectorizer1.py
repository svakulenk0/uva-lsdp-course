#
# Vectorizer.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy as np
import math
from collections import Counter
import pickle


class Vectorizer:
    def __init__(self, min_word_length=3, max_df=1.0, min_df=0.0):
        self.min_word_length    = min_word_length
        self.max_df             = max_df
        self.min_df             = min_df
        self.term_df_dict       = {}

    def fit(self, raw_documents):
        """Generates vocabulary for feature extraction. Ignores words shorter than min_word_length and document frequency
        not between max_df and min_df.
        :param raw_documents: list of string for creating vocabulary
        :return: None
        """
        self.document_count = len(raw_documents)
        doc_counts          = Counter([el for l in [Counter(doc.split()).keys() for doc in raw_documents] for el in l if len(el) >= self.min_word_length])

        for k, v in doc_counts.items():
            term_df = float(v + 1) / (self.document_count + 1)
            if term_df >= self.max_df or term_df <= self.min_df:
                continue
            self.term_df_dict[k] = term_df
        self.vocabulary = list(self.term_df_dict.keys())
        return

    def _transform(self, raw_document, method):
        """Creates a feature vector for given raw_document according to vocabulary.
        :param raw_document: string
        :param method: one of count, existance, tf-idf
        :return: numpy array as feature vector
        """
        feature_vec             = np.zeros(shape=len(self.get_feature_names()))
        doc_term_counts         = Counter(raw_document.split())
        # ------------------------------------------------------------- #
        if method == 'existance':       # <-- EXISTANCE METHOD
            for i, term in enumerate(self.get_feature_names()):
                if term in doc_term_counts.keys():
                    feature_vec[i] = 1
        # ------------------------------------------------------------- #
        elif method == 'count':         # <-- COUNT METHOD
            for i, term in enumerate(self.get_feature_names()):
                if term in doc_term_counts.keys():
                    feature_vec[i] = doc_term_counts[term]
        # ------------------------------------------------------------- #
        elif method == 'tf-idf':        # <-- TF-IDF METHOD
            for i, term in enumerate(self.get_feature_names()):
                if term in doc_term_counts.keys():
                    tf              = float(doc_term_counts[term])
                    idf             = math.log(1./self.term_df_dict[term]) + 1
                    feature_vec[i]  = tf*idf
            feature_vec_norm    = np.linalg.norm(feature_vec)
            if feature_vec_norm != 0:
                feature_vec /= feature_vec_norm  # L2 NORMALIZATION
        # ------------------------------------------------------------- #
        return feature_vec

    def transform(self, raw_documents, method="tf-idf"):
        """For each document in raw_documents calls _transform and returns array of arrays.
        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        return np.vstack([self._transform(doc, method) for doc in raw_documents])

    def fit_transform(self, raw_documents, method="tf-idf"):
        """Calls fit and transform methods respectively.
        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        self.fit(raw_documents)
        return self.transform(raw_documents, method)

    def get_feature_names(self):
        """Returns vocabulary.
        :return: list of string
        """
        try:
            self.vocabulary
        except AttributeError:
            print("Please first fit the model.")
            return []
        return self.vocabulary

    def get_term_dfs(self):
        """Returns number of occurances for each term in the vocabulary in sorted.
        :return: array of tuples
        """
        return sorted(self.term_df_dict.iteritems(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    def save(self):
        """Saves vectorizer to disk"""
        with open('../model/vectorizer1.p', 'wb') as outfile:
            pickle.dump((self.term_df_dict, self.vocabulary), outfile, pickle.HIGHEST_PROTOCOL)
        print("-> vectorizer is saved.")
        return

    def load(self):
        """Loads vectorizer from disk"""
        with open('Datasets/model/vectorizer1.p', 'rb') as infile:
            (self.term_df_dict, self.vocabulary) = pickle.load(infile)
        return
