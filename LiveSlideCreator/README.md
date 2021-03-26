# Live Slide Creation for the Perfect Lecture

This [Jupyter notebook](https://jupyter.org/) provides the codes and technical
report for our research project.  The goal of the research is to create a
pipeline from live speech as input to perfect slides as output. This is done
using NLP techniques like speech to text, named entity recognition and noun
chunk extraction. The slides are written to a markdown file and converted to
slides using the [lookatme](https://github.com/d0c-s4vage/lookatme) project.

## Folder Structure

The code and technical report are combined in the file final.ipynb.

```bash
.
├── README.md               # This file :)
├── data
│   ├── ner_dataset.csv     # Dataseet for NER
│   └── transcript.txt      # Transcript used for testing the presentation generator
├── docs
│   ├── lsg.png             # Graph for in the notebook
│   └── result.mp4          # Screencast of resulting presentation
├── final.ipynb             # The technical report and code
└── requirements.txt        # Lists requirements
```

## Requirements
- nltk
- noisereduce
- numpy
- pandas
- pyaudio
- spacy
- torch
- transformers

Install using `pip3 install -r requirements.txt`.

## Install and Run

Make sure you have a Jupyter Notebook kernel running. Then run by selecting
run all cells.

## Contributions

The contributions per team member are listed below:

- **Diego van der Mast**: Mainly worked on the live speech-to-text part and
  wrote the discussion and conclusion and the abstract.
- **Marvin Ong**: Worked mainly on named entity recognition (NER) and a little
  bit on noun phrase extraction.  Evaluated the NER model and also ensured
  that all sub-models work as a whole for the final slide creator model.
  Wrote the entire results section and the method and evaluation for NER.
- **Yochem van Rosmalen**: Mostly created the noun phrase extraction,
  summarizer and slide creator.  Wrote the Introduction section and the Method
  for the noun phase extraction and summarizer. Also wrote README.md.
