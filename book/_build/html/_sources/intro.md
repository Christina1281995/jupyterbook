## Overview

<br>

##### The Main Methods for Document-Level Sentiment Analysis

- Rule-Based / Lexicon Approaches

- ML approaches, e.g. Naive Bayes, SVM

- DL approaches, RNNs LSTMs and especially Transformer-based models

Rule-based lexicon approaches, as well as machine learning approaches have nowadays largely been overtaken in nearly every NLP task by deep learning models, especially Transformers. 
It's worth noting, that there are still robust methodologies in the machine learning domain, e.g. Naive Bayes or SVM classifiers, that achieve good results. Yet, they are outperformed in speed, scale, accuracy and robustness by the many deep learning architectures that have sprung up in recent years. 

Although there are various readily available libraries for machine learning methods (e.g. nltk provides an implementation of the Naive Bayes classifier, or scikit-learn offers an implementation of the SVM classifier), which are demonstrated in this demo, deep learning models are most often used for sentiment analysis tasks

Deep learning models like transformers have several advantages over traditional machine learning models:
- Better Representation Learning: Deep learning models can learn distributed representations of text data, which capture complex relationships and patterns in the data. This is particularly useful in NLP where words have multiple senses and the context in which words are used is important for determining their meaning.

- Large Scale Training: Deep learning models can be trained on large-scale text data, allowing them to learn from a wider variety of examples and achieve better performance.

- Improved Generalization: Deep learning models have the ability to generalize well to unseen data, making them well-suited for NLP tasks where the data distribution can be complex and hard to model.

- Attention Mechanisms: Transformers, in particular, have attention mechanisms that allow them to focus on the most relevant parts of the input data, making them well-suited for tasks such as sentiment-  classification where specific words and phrases can carry a lot of sentiment information.



```{tableofcontents}
```
