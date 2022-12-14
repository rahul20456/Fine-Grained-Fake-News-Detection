# Fine-Grained-Fake-News-Detection

Given a statement (tweet, political debate, ad etc), we predicted the
truthfulness rating of the statement -- pants-fire, false, barely true, half-true, mostly-true,
and true. This is a multi-label classification task.


Methodology

We preprocessed the data we needed, both training as well as testing, in the following manner: 
Lowercasing the text and any other metadata if we used and stemming using nltk library routine, 
PorterStemmer.

We used eight baselines to solve the multi-class
text classification problem. We used linear regression classifier (LR), a supervised support 
vector machine classifier (SVM), a decision Tree classifier (DT), a k-nearest neighbor classifier
(KNN), a bi-directional long short-term memory networks model (Bi-LSTMs), an artificial neural 
network, (ANN), and a distilBERT model. For LR, SVM, DT, RF, and KNN, we used the sklearn 
library. For ANN, and Bi-LSTMs, we used Tensor- Flow for the implementation. For distilBERT, we 
used PyTorch for the implementation.


We compared the results of the machine learning models, using two different settings. In the first
setting, we used the TF-IDF vectorizer to vectorize the corpus. In the second setting, we built 178
a more robust model. We used the Sentence Transformer, to convert the corpus into vectors.



experimental results 



![image](https://user-images.githubusercontent.com/88978808/207588896-224520c8-594e-4383-b76d-a010fcd5fddb.png)


analysis 


We observed that the F1 macro score for the machine learning models were less when we used  
only the attribute – ‘statements. However, when we added the meta-data: ‘subject’, ‘speaker’, and
the ‘party affiliation’, the F1 score increased. Thus,the model performs better on 
addition of more information as expected. Secondly, we observed that the F1 macro scores were less when we used the TF-IDF vectorizer for
vectorizing the data. However, when we used the sentence transformer to convert the corpus to the
vector space, the F1 score increased.


