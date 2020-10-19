import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from fastai import *
#from fastai.text import *
import os
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
#import seaborn as sns
predict_test=list()
df=pd.read_csv('/export/home/andrew/Bureau/SharedTask/malayalam_train.tsv', sep='\t', error_bad_lines=False)

#Data Exploration

col = ['category', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
df.columns = ['category', 'text']
df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)
print(df)

#imbalance classes
##fig = plt.figure(figsize=(8,6))
##df.groupby('category').text().plot.bar(ylim=0)
##plt.show()

#Text Representation
#using tfidf tokenizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id
print(features.shape)
#using chi2
N = 2
for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(category))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
#multiclass classifier features and design

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
df_t=pd.read_csv('tamil_test.tsv', sep='\t', error_bad_lines=False)
filename="tamil_output.tsv"
print("df_t:",len(df_t))
for i in range(len(df_t)):
    predict_test.append(clf.predict(count_vect.transform([df_t.iloc[i]["text"]]))[0])
#print(clf.predict(count_vect.transform(["Sirappana Tharamana Sambavagala Inimay than pakka pora"])))
print("predict_test:",type(predict_test[1]))
#df_t.assign(Name='label')
df_t["label"]=predict_test

#model selection
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV=5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
##sns.boxplot(x='model_name', y='accuracy', data=cv_df)
##sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
##              size=8, jitter=True, edgecolor="gray", linewidth=2)
##
##plt.show()
print(cv_df.groupby('model_name').accuracy.mean())
#df_t.to_csv(filename, sep='\t',index=False)
