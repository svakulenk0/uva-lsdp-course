# The good, the bad and the AI

# Table of Contents
- [Abstract](#Abstract)
- [Introduction](#Introduction)
- [Dependencies](#Dependencies)
- [Datasets](#Datasets)

# Abstract
- In this paper we use movie characters lines to analyse whether these can be used to train a model that is able to classify polar sentiments of the characters. Furthermore we will look into labeling movie genres based on either movie plot summaries or movie subtitles. Similar work on movie character sentiments has been done by studying relations between these characters (Lei Ding and Alper Yilmaz). The predictions of polar sentiments yield differing results, even though the models do not achieve high accuracy scores we believe that with future efforts these can be improved. The genre prediction of our pretrained model performed somewhat adequate and fell in line with our adjusted expectations after seeing the results of our first two tests.

# Introduction
- This report is made by Martijn van Raaphorst, Jens Ruhof and Luc Vink for the course Language, speech and dialogue processing, Bsc K.I. at the University of Amsterdam. It was supervised by Iris Lau, and Svitlana Vakulenko.

# Dependencies
- python 3.8 or lower
	- pandas
	- sklearn
    - tensorflow 1.15.0 (requires 3.8 or lower)
    - keras 2.3.1 
    - LIME
    - seaborn
    - convokit

# Datasets
The datasets as we use them can be downloaded from:
- [Cornell](https://convokit.cornell.edu/documentation/movie.html)
- [CMU](http://www.cs.cmu.edu/~ark/personas/)
- [cetinsamet's Movie subtitles dataset+models](https://github.com/cetinsamet/movie-genre-classification)
- [British Political Speech database](http://www.britishpoliticalspeech.org/speech-archive.htm)