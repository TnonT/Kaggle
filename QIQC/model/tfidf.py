# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-19下午9:30
# @file:tfidf.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm
from sklearn.metrics import accuracy_score,recall_score,roc_curve

train_path = '../data/train.csv'
# test_path = '../data/test.csv'

train_df = pd.read_csv(train_path).head(55000)
# test_df = pd.read_csv(test_path)

train_df.dropna()

X = train_df.question_text
Y = train_df.target

VOCAB_SIZE = 13500
counter = CountVectorizer(max_features=VOCAB_SIZE)
Xc = counter.fit_transform(X)
tf_idfvec = TfidfTransformer()
X_ = tf_idfvec.fit_transform(Xc)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_,Y,test_size=0.2,random_state=0)

clf = svm.SVC()

clf.fit(X_train, Y_train)
Y_dev_pre = clf.predict(X_dev)

acc = accuracy_score(Y_dev, Y_dev_pre)
print("The acc is {}".format(acc))
rec_score = recall_score(Y_dev, Y_dev_pre)
print("The recall socre is {}".format(rec_score))

fpr, tpr, thresholds = roc_curve(Y_dev, Y_dev_pre, pos_label=2)
print("fpr is {0}, tpr is {1}, thresholds is {2}".format(fpr,tpr,thresholds))
