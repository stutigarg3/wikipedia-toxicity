# Objective:  
The Wikipedia Talk Labels dataset is a collection of comments from Wikipedia's talk pages that have been labeled as either "toxic" or "not toxic". The comments were collected from 2015-2016 and cover a wide range of topics. The dataset was created as part of the Wikipedia Detox project, which aims to reduce harassment and increase civility on Wikipedia. The main objective of our project is to use this dataset for text classification tasks related to hate speech detection and toxic comment classification. 

# Data Set Description: 
The Wikipedia Talk Labels dataset is a collection of around 160,000 comments from Wikipedia's talk pages, which have been labeled as either toxic or not toxic. The data is intended to be used for tasks related to hate speech detection and toxic comment classification. The dataset can be found on the figshare platform, and it includes a CSV file containing the comments along with their labels. The data has been labeled by human annotators using the Crowdflower platform, and it includes additional features such as worker demographic information and a label indicating the level of consensus among the annotators. The dataset is suitable for use in machine learning applications, as well as for research into online toxicity and hate speech. The dataset consists of 2 files, one with the text and other with the annotations. The files have 160,000 rows and 6 columns for text file and 4 columns for the annotations file.

# Preliminary Data Exploration: 

# Summary Statistics:

Total number of comments: 159,686
Number of toxic comments: 16,087 (10.07%)
Number of non-toxic comments: 143,599 (89.93%)
Average length of comments (in characters): 297.7
Minimum length of comments (in characters): 6
Maximum length of comments (in characters): 5,466


Visualization:

						Fig.1
Fig.1: Histogram of comment lengths: A histogram of comment lengths shows that the distribution is positively skewed, with a long tail towards the right. The majority of comments are relatively long, with a peak around 750-1000 words in length. However, there are also a significant number of comments that are much longer, with some exceeding 5,000 characters.


# Something Interesting:
The distribution of toxic and non-toxic comments is heavily imbalanced, with non-toxic comments making up the vast majority of the dataset. This can present challenges when building machine learning models. In this case, it might be necessary to use techniques such as oversampling, undersampling, or class weighting to address the imbalance and ensure that the model can accurately detect both toxic and non-toxic comments.

# Predictions:
We will use the Wikipedia Talk Labels dataset to make predictions about whether a given comment from Wikipedia's talk pages is toxic or not. Specifically, the dataset can be used for hate speech detection and toxic comment classification tasks, where the goal is to train a machine learning model to accurately predict whether a new comment is toxic or not.

More specifically, using this dataset, we will perform classification tasks, where we will train a machine learning model to classify the comments as toxic or non-toxic. We will use different algorithms and techniques such as logistic regression, support vector machines (SVM), random forests, or deep learning-based models to make these predictions.

# Inference:
1. The predictions made using this dataset can be useful for various applications, including identifying and moderating toxic comments on Wikipedia or other online platforms, improving online conversation quality, and preventing or mitigating online harassment and abuse.
2. Develop a machine learning model that accurately classifies comments from Wikipedia's talk pages as toxic or non-toxic, and evaluate its performance using various metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
3. Analyze the prevalence and distribution of toxic comments across different categories such as race, gender, religion, or political affiliation, and identify any patterns or biases that might exist.
4. Explore the relationships between toxic comments and various social, cultural, or political factors such as media coverage, public opinion, or policy changes, and identify any trends or correlations that might exist.
5. Compare the performance of different machine learning algorithms and techniques for hate speech detection and toxic comment classification tasks, and evaluate their robustness and generalizability to other datasets or domains.

# Non Spark Packages: 
Scikit learn, Tensor flow, Keras.

For this text classification task, we will be using TensorFlow and the Keras API to build sequential models for classifying text. To improve the performance of our models, we plan to use hyperparameter tuning to explore multiple architectures and maximize the F1 score, particularly given the unbalanced nature of our dataset. Additionally, we will use Scikit-Learn to build standard text classification algorithms, such as Naive-Bayes and SVM. This will allow us to compare the performance of our deep learning models with traditional machine learning algorithms and determine which approach works best for our specific task.


 MODEL                    ACCURACY                 AUC ROC
Logistic Regression        78.7%                     0.57
Naive Bayes                88.9%                     0.75
Support Vector Machine     83.25%                    0.56

