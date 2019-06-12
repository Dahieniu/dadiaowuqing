from time import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
np.set_printoptions(linewidth=400)

# 步骤1：载入新闻语料文档
print("loading train dataset ...")
t = time()
# load file 专门用于载入分类的文档，每个分类一个单独的目录，目录名就是类名
news_train = load_files('E:/DATA_t')
print("summary: {0} documents in {1} categories.".format(
    len(news_train.data), len(news_train.target_names)))
# news_train.target是长度为13180的一维向量，每个值代表相应文章的分类id
print(r'news_categories_names:\n{}, \nlen(target):{}, target:{}'.format(news_train.target_names,
                                                                       len(news_train.target), news_train.target))
print(r"done in {0} seconds\n".format(round(time() - t, 2)))

print("vectorizing train dataset ...")
t = time()
vectorizer = TfidfVectorizer(encoding='latin-1')
X_train = vectorizer.fit_transform((d for d in news_train.data))
print("n_samples: %d, n_features: %d" % X_train.shape)
# X_train每一行代表一篇文档，每个成员表示一个词的TF-IDF值，表示这个词对这个文章的重要性。
# X_train的形状是13180X130274
print("number of non-zero features in sample [{0}]: {1}".format(
    news_train.filenames[0], X_train[0].getnnz()))
print(r",, in {0} seconds\n".format(round(time() - t, 2)))

print("traning models ...".format(time() - t))
t = time()
y_train = news_train.target
clf = BernoulliNB(alpha=0.0001)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print("train score: {0}".format(train_score))
print(r"done in {0} seconds\n".format(round(time() - t, 2)))
print("loading test dataset ...")
t = time()
news_test = load_files('E:/TEST')
print("summary: {0} documents in {1} categories.".format(
    len(news_test.data), len(news_test.target_names)))
print("done in {0} seconds".format(time() - t))
print("vectorizing test dataset ...")
t = time()
X_test = vectorizer.transform((d for d in news_test.data))
y_test = news_test.target
print("n_samples: %d, n_features: %d" % X_test.shape)
print("number of non-zero features in sample [{0}]: {1}".format(
    news_test.filenames[0], X_test[0].getnnz()))
print("done in %fs" % (time() - t))
pred = clf.predict(X_test[0])
print("predict: {0} is in category {1}".format(
    news_test.filenames[0], news_test.target_names[pred[0]]))
print("actually: {0} is in category {1}".format(
    news_test.filenames[0], news_test.target_names[news_test.target[0]]))
print("predicting test dataset ...")
t0 = time()
pred = clf.predict(X_test)
print("done in %fs" % (time() - t0))
from sklearn.metrics import classification_report

print("classification report on test set for classifier:")
print(clf)
print(classification_report(y_test, pred,
                            target_names=news_test.target_names))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
print("confusion matrix:")
print(cm)

# Show confusion matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8), dpi=144)
plt.title('Confusion matrix of the classifier')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.matshow(cm, fignum=1, cmap='gray')
plt.colorbar()
plt.show()




