
import pandas as pd
import hazm
import nltk
from nltk.stem.porter import *
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
import sys
import os
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('divar_posts_dataset.csv', encoding='utf-8')
result_path = 'result'
if not os.path.exists(result_path):
    os.mkdir(result_path)


def isEnglish(word):
    try:
        word.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_stopword():
    filepath = 'persian_stopwrod'
    persian_stopwrod =[]
    with open(filepath, mode='r', encoding='utf-8-sig') as fp:
       line = fp.readline()
       while line:
           persian_stopwrod.append(line.strip())
           line = fp.readline()

    stop_words = set(stopwords.words('english'))
    stop_words.union(persian_stopwrod)
    return stop_words

#section one: predict brand mobile
section_one_result = open('result/section_one.txt', "w")

data_mobile = dataset[dataset['cat3'] == "mobile-phones"]
section_one_result.write('size of data mobile in dataset: %s\n' %str(data_mobile.shape))

NUM = data_mobile.shape[0]
# NUM = 3000

#preprovessing
corpus = ["" for x in range(NUM)]
target = ["" for x in range(NUM)]

stemmer = PorterStemmer()
normalizer = hazm.Normalizer()
stemmer_far = hazm.Stemmer()
stop_words = get_stopword()
for i in range(NUM):
    if i % 1000 ==0:
        sys.stdout.write("\rDoing preprocessing %i/ %i " % (i, NUM))
        sys.stdout.flush()
    item = data_mobile.iloc[i]["title"] + ' ' + data_mobile.iloc[i]["desc"]
    item =  nltk.word_tokenize(item)
    item =  [w for w in item if not w in stop_words] #delete stop word in english and farsi language
    item = np.array(item)

    index_english =  list(map(isEnglish, item))
    index_farsi =  np.logical_not(index_english)
    index_english = np.where(index_english)

    en_token = item[index_english]
    en_token =  list(map(stemmer.stem, en_token))

    fa_token = item[index_farsi]
    fa_token = list(map(normalizer.normalize, fa_token))
    fa_token = list(map(stemmer_far.stem, fa_token))

    corpus[i] = en_token + fa_token  #combine farsi and english token
    target[i] = data_mobile.iloc[i]["brand"]


#creat bag of word from english and farsi token
vectorizer = CountVectorizer(tokenizer=lambda s:s,lowercase=False, min_df=2, max_df=0.6)
data = vectorizer.fit_transform(corpus)
data =  data.toarray()
print("\nsize of feature(bag of word):" , data.shape[1])
section_one_result.write("\nsize of feature(bag of word): %i \n" %data.shape[1])

#encode target(brand) in numerical number
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)
section_one_result.write("section one: predict brand mobile\n")
section_one_result.write("brand exist in dataset:  %s\n" %list(le.classes_))
section_one_result.write("number of brand exist in dataset:  %i\n\n" % ( len(list(le.classes_))))

gnb = GaussianNB()
clf_LR = LogisticRegression(random_state=0, solver='sag', multi_class='ovr',max_iter=100,n_jobs=9,class_weight='balanced',verbose=1)
NUMBER_FOLD = 4
kf = KFold(n_splits=NUMBER_FOLD)
acc_gaussianNB = np.zeros((NUMBER_FOLD,2))
f1_gaussianNB = np.zeros((NUMBER_FOLD,2))

acc_LogisticRegression = np.zeros((NUMBER_FOLD,2))
f1_LogisticRegression=np.zeros((NUMBER_FOLD,2))
i=0
for train, test in kf.split(data):
    X_train, X_test, y_train, y_test = data[train], data[test], target[train], target[test]
    print(i ,'GNB ')

    model_gaussianNB= gnb.fit(X_train, y_train)

    print(i, ' pred train')
    y_train_pred = model_gaussianNB.predict(X_train)
    acc_gaussianNB[i,0] = metrics.accuracy_score(y_train, y_train_pred, normalize=True)
    f1_gaussianNB[i,0] =metrics.f1_score(y_train, y_train_pred, average='weighted', labels=np.unique(y_train_pred))

    print(i, ' pred test')
    y_test_pred = model_gaussianNB.predict(X_test)
    acc_gaussianNB[i,1] = metrics.accuracy_score(y_test, y_test_pred, normalize=True)
    f1_gaussianNB[i, 1] = metrics.f1_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred))

    print(i ,'LR')
    model_LogisticRegression= clf_LR.fit(X_train, y_train)

    print(i ,' LR pred train')
    y_train_pred = model_LogisticRegression.predict(X_train)
    acc_LogisticRegression[i,0] = metrics.accuracy_score(y_train, y_train_pred, normalize=True)
    f1_LogisticRegression[i,0] =metrics.f1_score(y_train, y_train_pred, average='weighted', labels=np.unique(y_train_pred))

    print(i ,' LR pred test')
    y_test_pred = model_LogisticRegression.predict(X_test)
    acc_LogisticRegression[i,1] = metrics.accuracy_score(y_test, y_test_pred, normalize=True)
    f1_LogisticRegression[i, 1] = metrics.f1_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred))

    section_one_result.write("fold: %i naive bayes, accuracy for train:%f, accuracy for test:%f, f1 for train:%f, f1 for test:%f\n"
                              % (i+1,acc_gaussianNB[i,0], acc_gaussianNB[i,1], f1_gaussianNB[i,0], f1_gaussianNB[i,1]))

    section_one_result.write("fold: %i Logistic regression, accuracy for train:%f, accuracy for test:%f, f1 for train:%f, f1 for test:%f\n"
                              % (i+1,acc_LogisticRegression[i,0], acc_LogisticRegression[i,1], f1_LogisticRegression[i,0], f1_LogisticRegression[i,1]))

    i += 1

mean_acc_gaussianNB = np.mean(acc_gaussianNB,axis=0)
mean_f1_gaussianNB = np.mean(f1_gaussianNB,axis=0)


mean_acc_LogisticRegression = np.mean(acc_LogisticRegression,axis=0 )
mean_f1_LogisticRegression = np.mean(f1_LogisticRegression,axis=0 )

section_one_result.write(
    "\n\nNaive bayes avergae:  accuracy for train:%f, accuracy for test:%f, f1 for train:%f, f1 for test:%f\n"
    % (mean_acc_gaussianNB[0], mean_acc_gaussianNB[1], mean_f1_gaussianNB[ 0],
       mean_f1_gaussianNB[1]))

section_one_result.write(
    "Logistic regression avergae:  accuracy for train:%f, accuracy for test:%f, f1 for train:%f, f1 for test:%f\n"
    % (mean_acc_LogisticRegression[0], mean_acc_LogisticRegression[1], mean_f1_LogisticRegression[ 0],
       mean_f1_LogisticRegression[1]))