# sarcastic-comment-detection

## Introduction

```Sarcasm is the caustic use of irony, in which words are used to communicate the opposite of their surface meaning, in a humorous way or to mock someone or something``` — [wikipedia](https://en.wikipedia.org/wiki/Sarcasm)

```Sarcasm is the use of words that mean the opposite of what you really want to say especially in order to insult someone, to show irritation, or to be funny``` — [merriam-webster](https://www.merriam-webster.com/dictionary/sarcasm)

Sarcasm is used in the news headlines to mock/criticize the Industries/MNCs/Government/Government policies. Sometimes news publishing houses intentionally use sarcastic headlines to sound funny or get the general public’s attention.

In this blog, BoW, TF-IDF text processing techniques are used followed by ML algorithms for classification. In another approach, pre-trained embedded vectors are used to encode words and Neural networks are used for classification. BERT encodings are also used but the results are very poor. The metric used for evaluation is accuracy. Finally, the best model is deployed on the Heroku cloud.

## [Link](https://binginagesh.medium.com/identifying-sarcastic-headlines-cf4a7c2382f9) to medium blog

## Results

| Model        | Test Accuracy           |
| ------------- |:-------------:|
| BoW with Logistic Regression    | 91.3287% |
| BoW with Naive Bayes      | 86.9557%      |
| BoW with Random Forest | 99.8951%      |
|  BoW with XGBoost    | 87.9815% |
| TF-IDF with Logistic Regression      | 90.7671%      |
| TF-IDF with Naive Bayes | 87.0979%      |
| TF-IDF with Random Forest    | 99.9063% |
| TF-IDF with XGBoost      | 91.1752%      |
| **Neural network with Glove Embeddings** | **99.9925%**      |


## [Link](https://sarcastic-comment.herokuapp.com/) to Heroku web-app

## Sample predictions

https://user-images.githubusercontent.com/46963154/141278774-adee3526-56df-44eb-a88e-c89ba17ec220.mp4

## Conclusion
Different experiments are done to classify sarcastic news headlines from non-sarcastic news headlines. The Random Forest model worked well for both BoW and TF-IDF featurization. The best accuracy was achieved with the neural network with Glove embeddings with an accuracy of **99.9925%**.
