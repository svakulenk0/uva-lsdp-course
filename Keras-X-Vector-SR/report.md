Speaker Recognition using Keras
Aleksej Horvat
Student Number: 10688536

---

![Foto by Jason Rosewell on Unsplash](https://images.unsplash.com/photo-1453738773917-9c3eff1db985?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80)

## Introduction

Speaker Recognition is a biometric task that allows an automated system to identify a speaker by creating a model of the features of a speakers utterances. Speaker recognition is applicable to the domains of security (*speaker verification*), informatiotion retreaval (*speaker identification*) and have common applications in todays smart devices or voice activated services. A smart device could consider the context of a request better if it is able to recognize which user has issued the command.

Diarization (segmentation)

"Hey [ASSISTANT], call Mom" - whose mom? which user's adressbook should be queried?
"OK [ASSISTANT], play my workout mix" - user's playlist?
"Hey [ASSISTANT]" (command issued by an unauthorised user) - system refuses to process command!

While early work on this topic focused on Gaussian Mixture Models, recently attention has shifted toward the use of Deep Neural Networks. i-vectors, a dimentioanl reduction approach, have been the base of many speaker recognition systems and are descibed in **Approaches**. [@Synder2018x] X-vectors also reduce the high dimensional feature space, but utilize a DNN discriminate between speakers as apposed to PLDA's or GMM in the case of i-vectos. An added benefet is that x-vectors models only need require speaker labels on training data, while i-vectors require transcribed data.

## Research Questions:


### Which characteristics of Dialogue (Speech) can be used to recognise individual agents?

### What is the accuracy of model?

### How does the model compare to existing methods?

### To what extent can speaker recognition be modelled using high-level instructions?




## Approaches to Speaker Recognition

Speaker recognition models fall into two categories, *text-dependant Speaker Verification* and *text-independant speaker verification*. In the former training data must be trancribed, whereas in the case of the latter there are no lexigraphical contrainsts imposed. The has the benefit of exposing a larger variance and durtion of potential training data. This research focusses on the latter.

The general approach to SR is to extract speaker embeddings from utterances (high dimentional space) and compare them using some distance funtion. The inherrant problem with this high dimensional space is the vector length, the data rate.

Both i-vectors and x-vectors map variable length utterances to fixed legth vectors.

In neural network sustems, the ouputs of the neural network are the embeddings (or d-vectors)

Feature Extraction
Sliding Window


![i vs x-vectors](img/i-x-vec.jpg) [@KellyF-et-al]

### I-vector
- GMM guassian mixture model
- UBM universal background model
- maximize ${\mathcal{L}}$ (likelihood)

### X-vectors
- state-of-the-art approach
- incorates temperal context (TDNN layers)
- x-vector models can capitalise on larger training sets. i-vector models tend to max out after a certain amount of training data and suffer diminishing returns with increased training data
- no need to transcribe data, lexicon free
- possible to augment datasets using noise leading to increased accuracy

<!-- ### D-Vectors -->

### PlDA

probabilistic linear discriminant analysis

## Dataset

The consulstanted works do not stress the signfigance of the particular dataset. Of immortance is a varied set of speakers, and the presence of noise. The trained model is subject to a high degree of error in the presence of a high signal-noise-ratio if was not trained on data sufficient noise. Accuracy can also vary when sampling data from different recording devices, using different compression or under varying auditory spaces, which the model may interpret as noise. Therefore it is beneficial to have sufficient variance in training data.

It was innitally planned to use the Spotify Podcast Dataset, but the absence of speaker labels renders this dataset unsuitable for x-vector training. Other candidates include "NIST speaker recognition evaluation 2016" and "Speakers in the Wild Core". The Keras model is trained using the Kaggle Speaker Recognition Dataset, containing speaches of prominent leaders. The pretrained models are trained on VoxCeleb and LDC corpora.

## Approach

The approach is create an x-vector model using the Keras framework. The model will be trained on available data and compared with the pretrained models performance.

### Keras

![keras](https://keras.io/img/logo.png) + ![Tensor Flow](https://www.tensorflow.org/images/tf_logo_social.png)

Keras is a high level framework for deep neural networks
- runs on top of other frameworks (Tensor Flow,  etc)
- less code needed to describe network / model
- framework specific concepts are abstracted (focus on the model, not the framework)
  - higher abstraction than TF
- integrated in TF 2.0 (can use same pipeline, tweak as needed)

## Evaluation
 - metrics **EER**

$$
\begin{align}
R_{FA} &= R_{FR} \\
R_{FA} &= \frac{\textrm{Number of False Acceptances}}{\textrm{Number of Impostor Acessess}} \\
R_{FR} &= \frac{\textrm{Number of False Rejections}}{\textrm{Number of target Acessess}}
\end{align}
$$

LDA

 - results
- error analysis

## Findings
- illustration
- interpretation
- discussion

## Conlusion
- summary
- lessons learned
- directions for future work
    - diarization (unsupervised)


## References:

Kelly, F., Alexander, A., Forth, O., & van der Vloed, D. From i-vectors to x-vectors–a generational change in speaker recognition illustrated on the NFI-FRIDA database.

Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018, April). X-vectors: Robust dnn embeddings for speaker recognition. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5329-5333). IEEE.

Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2018, April). Generalized end-to-end loss for speaker verification. In *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 4879-4883). IEEE.

Ma, B., Meng, H. M., & Mak, M. W. (2007, April). Effects of device mismatch, language mismatch and environmental mismatch on speaker verification. In *2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07* (Vol. 4, pp. IV-301). IEEE.