# Dutch Parliament Talks
## Description
With upcoming elections in mind, this notebook aims to show how NLP can be used to provide information about the Dutch political landscape. Especially regarding the parties that were included in the house of representatives. With 4 different research questions we hope to provide an extensive overview of sentiment, topics and framing.

**Our goal**: To turn diverse parliamentary data into interpretable resources.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages:

```sh
pip install pattern
pip install regex
pip install elementpath
pip install pandas
pip install matplotlib
pip install --user -U nltk
pip install seaborn
pip install IPython
pip install operator
pip install sklearn
pip install datetime
pip install gensim
```

## Usage

```python
import xml.etree.ElementTree as ET
import os
from pattern.nl import sentiment
import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt 
from nltk.corpus import wordnet
from pattern.nl import pluralize
from IPython.display import display_html
import seaborn as sns
nltk.download('wordnet')
nltk.download('omw')
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from datetime import date
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim import models
from pattern.nl import pluralize
import string
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
```

## Authors and acknowledgment
Authors:
Quinten Bolding
Marten Rozema
Julotte van der Beek
Savina Rielova

Special thanks to Ruben van Heusden, PhD student at the IRlab, for helping and providing us with the ParlaMint data and special thanks to Iris Lau, TA for the course Language Speech and Dialogue Processing, for helping us with weekly feedback!

