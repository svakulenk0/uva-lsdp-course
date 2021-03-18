#
# Preprocessor.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import os
from nltk.corpus import stopwords
import codecs
import errno
import string


class Preprocessor:
    def __init__(self, dataset_directory="Dataset", processed_dataset_directory= "ProcessedDataset"):
        self.dataset_directory = dataset_directory
        self.processed_dataset_directory = processed_dataset_directory
        nltk.download("stopwords",quiet=True)
        nltk.download("punkt",quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def _remove_puncs_numbers_stop_words(self, tokens):
        """Remove punctuations in the words, words including numbers and words in the stop_words list.
        :param tokens: list of string
        :return: list of string with cleaned version
        """
        tokens          = [token.replace("'", '') for token in tokens]
        tokens_cleaned  = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens_cleaned

    def _tokenize(self, sentence):
        """Tokenizes given string.
        :param sentence: string to tokenize
        :return: list of string with tokens
        """
        sentence_tokenized  = nltk.tokenize.word_tokenize(sentence.replace("\n", " "))
        return [token.lower() for token in sentence_tokenized]

    def _stem(self, tokens):
        """Stems the tokens with nltk SnowballStemmer
        :param tokens: list of string
        :return: list of string with words stems
        """
        stemmer         = nltk.SnowballStemmer(language='english')
        tokens_stemmed  = [stemmer.stem(token) for token in tokens]
        return tokens_stemmed

    def preprocess_document(self, document):
        """Calls methods _tokenize, _remove_puncs_numbers_stop_words and _stem respectively.
        :param document: string to preprocess
        :return: string with processed version
        """
        doc_tokenized   = self._tokenize(document)
        doc_cleaned     = self._remove_puncs_numbers_stop_words(doc_tokenized)
        doc_stemmed     = self._stem(doc_cleaned)
        doc_stemmed_str = ' '.join(doc_stemmed)
        return doc_stemmed_str

    def preprocess(self):
        """Walks through the given directory and calls preprocess_document method. The output is
        persisted into processed_dataset_directory by keeping directory structure.
        :return: None
        """
        for root, dirs, files in os.walk(self.dataset_directory):
            if os.path.basename(root) != self.dataset_directory:
                print("Processing", root, "directory.")
                dest_dir = self.processed_dataset_directory + "/" + root.lstrip(self.dataset_directory + "/")
                if not os.path.exists(dest_dir):
                    try:
                        os.makedirs(dest_dir)
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                for file in files:
                    file_path = root + "/" + file
                    with codecs.open(file_path, "r", "ISO-8859-1") as f:
                        data = f.read().replace("\n", " ")
                    processed_data = self.preprocess_document(data)
                    output_file_path = dest_dir + "/" + file
                    with codecs.open(output_file_path, "w", "ISO-8859-1") as o:
                        o.write(processed_data)
