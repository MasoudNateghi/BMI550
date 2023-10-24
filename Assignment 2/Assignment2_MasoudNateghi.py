#%% import libraries
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from sklearn import svm
import xgboost as xgb

#%% read data
train_data = pd.read_csv('fallreports_2023-9-21_train.csv')
test_data = pd.read_csv('fallreports_2023-9-21_test.csv')

# remove fall descriptions with NaN values
train_indices = ~train_data['fall_description'].isna()
test_indices = ~test_data['fall_description'].isna()

# extract texts from data
train_texts = list(train_data['fall_description'][train_indices])
test_texts = list(test_data['fall_description'][test_indices])

# extract labels
trainy = list(train_data['fog_q_class'][train_indices])
testy = list(test_data['fog_q_class'][test_indices])

#%% Word2Vec pre-trained models
import gensim.downloader as api
print(list(api.info()['models'].keys()))

wv1 = api.load('word2vec-google-news-300')
wv2 = api.load('glove-wiki-gigaword-300')

#%% generate different selection combinations for features
# 0: n-gram count vector
# 1: cluster count vector
# 2: n-gram count tf-idf
# 3: cluster count tf-idf
# 4: Word2Vec
# 5: gLoVe

from itertools import combinations
# store different selction combinations
feature_index = []
for i in range(6+1):
    if i == 0: continue
    combination_length = i  
    
    # Get all combinations of the specified length from the list
    combs = list(combinations(range(6), combination_length))
    
    # Print the index of all selection combinations
    for comb in combs:
        feature_index.append(comb)
        
#%% Preprocess Stage
# import stemmer
stemmer = PorterStemmer()

# import stopwords
sw = stopwords.words('english')

def preprocess_text(raw_text):
    #Lowercase
    text = raw_text.lower()
    #word tokenize
    tokenized = word_tokenize(text)
    #apply a stemmer
    words = [stemmer.stem(st) for st in tokenized if not st in ['.',',']] # and not st in sw (to be added)
    return (" ".join(words))

def loadwordclusters():
    infile = open('./50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

def w2v (texts, model):
    # receives a list of sentences and embedding model: Word2Vec, gLoVe as input
    # outputs word2vec representation for each sentence in an array of size n*d 
    # where n: number of sentences, d: dimension of representation in w2v model
    
    # number of sentences
    n = len(texts)
    # dimension of representation
    d = 300
    # w2v representation
    w2v_rep = np.zeros((n, d)) 
    for i in range(n):
        # word tokenize
        words = word_tokenize(texts[i])
        # calculate mean w2v representatoin for the words of the sentence
        w2v_rep[i, :] = np.mean([model[word] for word in words if word in model], axis=0)
    return w2v_rep

# load word clusters
word_clusters = {}
word_clusters = loadwordclusters()

# do preprocessings
train_texts_preprocessed = []
test_texts_preprocessed = []
train_texts_cluster = []
test_texts_cluster = []

for i in range(len(train_texts)):
    train_texts_preprocessed.append(preprocess_text(train_texts[i]))
    train_texts_cluster.append(getclusterfeatures(train_texts[i]))
for i in range(len(test_texts)):
    test_texts_preprocessed.append(preprocess_text(test_texts[i]))
    test_texts_cluster.append(getclusterfeatures(test_texts[i]))

#%% extracting features
# vectorizers
count_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000) # n-gram count vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000) # tf-idf vectorizer
scaler = StandardScaler()

# count vector on n-grams
trainx_cv_ngram = count_vectorizer.fit_transform(train_texts_preprocessed).toarray()
testx_cv_ngram = count_vectorizer.transform(test_texts_preprocessed).toarray()

# count vector on cluster
trainx_cv_cluster = count_vectorizer.fit_transform(train_texts_cluster).toarray()
testx_cv_cluster = count_vectorizer.transform(test_texts_cluster).toarray()

# tf-idf on n-gram
trainx_tfidf_ngram = tfidf_vectorizer.fit_transform(train_texts_preprocessed).toarray()
testx_tfidf_ngram = tfidf_vectorizer.transform(test_texts_preprocessed).toarray()

# tf-idf on cluster
trainx_tfidf_cluster = tfidf_vectorizer.fit_transform(train_texts_cluster).toarray()
testx_tfidf_cluster = tfidf_vectorizer.transform(test_texts_cluster).toarray()

# w2v 
trainx_w2v = w2v(train_texts_preprocessed, wv1)
testx_w2v = w2v(test_texts_preprocessed, wv1)

# glove 
trainx_glove = w2v(train_texts_preprocessed, wv2)
testx_glove = w2v(test_texts_preprocessed, wv2)


# normalization
trainx_cv_ngram = scaler.fit_transform(trainx_cv_ngram)
testx_cv_ngram = scaler.transform(testx_cv_ngram)
trainx_cv_cluster = scaler.fit_transform(trainx_cv_cluster)
testx_cv_cluster = scaler.transform(testx_cv_cluster)
trainx_tfidf_ngram = scaler.fit_transform(trainx_tfidf_ngram)
testx_tfidf_ngram = scaler.transform(testx_tfidf_ngram)
trainx_tfidf_cluster = scaler.fit_transform(trainx_tfidf_cluster)
testx_tfidf_cluster = scaler.transform(testx_tfidf_cluster)
trainx_w2v = scaler.fit_transform(trainx_w2v)
testx_w2v = scaler.transform(testx_w2v)
trainx_glove = scaler.fit_transform(trainx_glove)
testx_glove = scaler.transform(testx_glove)

# store all features inside a list
trainx_all = [trainx_cv_ngram, trainx_cv_cluster, trainx_tfidf_ngram, trainx_tfidf_cluster, trainx_w2v, trainx_glove]
testx_all = [testx_cv_ngram, testx_cv_cluster, testx_tfidf_ngram, testx_tfidf_cluster, testx_w2v, testx_glove]

#%% hyperparameter tuning
# function to find hyperparameters using grid search with 5-Fold CV
def grid_search_hyperparam_space(params, classifier, x_train, y_train):
        grid_search = GridSearchCV(estimator=classifier, param_grid=params,
                                   refit=True, cv=5, return_train_score=False,
                                   scoring='accuracy', verbose=2)
        grid_search.fit(x_train, y_train)
        return grid_search

def train_grid(x, y, model, hyperparams):
    # store best peformance of each model with optimal hyperparameters for each
    #combination of feature sets
    performance = []
    # iterate ove each combination of featuresets 
    for sel in feature_index:
        # create the respective feature set
        trainx = x[sel[0]]
        i = 1
        while(len(sel) > i):
            trainx = np.concatenate((trainx, x[sel[i]]), axis=1)
            i += 1
        # use gridsearch to find hyperparameters for the given combination of feature sets 
        grid = grid_search_hyperparam_space(hyperparams, model, trainx, y)
        print('Feature Sets: ', sel)
        print('Best hyperparameters: ', grid.best_params_)
        print('Score: ', grid.best_score_)
        # store performance
        performance.append([sel, grid.best_params_, grid.best_score_])
    return performance

# for more complex models like XGBoost or RF which have lots of hyperparameters
# and the training process is time-consuming we use this function which basically
# uses optimal feature sets where we have previously found for more simple models
# like logistic regression, knn, and etc. Using these optimal features, this function
# uses 5-fold CV to find the optimal hyperparameters. 
def train_grid_big_model(x, y, model, hyperparams):
    # perform grid search
    grid = grid_search_hyperparam_space(hyperparams, model, x, y)
    print('Best hyperparameters')
    print(grid.best_params_)
    print('All scores:')
    all_means = grid.cv_results_['mean_test_score']
    all_standard_devs = grid.cv_results_['std_test_score']
    all_params = grid.cv_results_['params']
    for mean, std, params in zip(all_means, all_standard_devs, all_params ):
        print(mean, std, params)
        
# define models
# SVM
model1 = svm.SVC()

# XGBoost 
model2 = xgb.XGBClassifier(
    n_estimators=100,  # Number of boosting rounds (trees)
    max_depth=3,       # Maximum depth of each tree
    learning_rate=0.1,  # Step size shrinkage to prevent overfitting
    objective='binary:logistic'  # For binary classification
)

# Random Forrest
model3 = RandomForestClassifier(random_state=42)

# Logistic Regression
model4 = LogisticRegression(max_iter=1000)

# KNN
model5 = KNeighborsClassifier()

# determine hyperparameters of each model
# SVM
grid_params_model1 = {
    'C': [0.1, 1, 2, 5, 10],
    'gamma': [1, 0.1, 0.01, 0.001], 
    'kernel':['rbf', 'linear'] 
}

# XGBoost
grid_params_model2 = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Random Forrest
grid_params_model3 = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Logistic Regression
grid_params_model4 = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# KNN
grid_params_model5 = {'n_neighbors': [3, 5, 7, 9]}
#%% find optimal feature set combo
# given the performance of the model for each combination of feature sets
# this function finds best set of features and hyperparameters
def find_best_sel_hyp(result):
    best_acc = -1
    best_index = 0
    for i in range(len(result)):
        if result[i][2] >= best_acc: 
            best_acc = result[i][2]
            best_index = i
    
    print('Feature Sets: ', result[best_index][0])
    print('Best hyperparameters: ', result[best_index][1])
    print('Score: ', result[best_index][2])
#%% SVM
# Monitor Performance of eacg model
print('SVM Classifier: ') 
result = train_grid(trainx_all, trainy, model1, grid_params_model1)
print('-------------------------------------------')
find_best_sel_hyp(result)
# Feature Sets:  (4, 5)
# Best hyperparameters:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
# Score:  0.8684180790960452

#%% Logistic Regression
print('Logistic Regression Classifier: ') 
result = train_grid(trainx_all, trainy, model4, grid_params_model4)
find_best_sel_hyp(result)
# Feature Sets:  (2, 4, 5)
# Best hyperparameters:  {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
# Score:  0.8514689265536723
#%% KNN
print('KNN Classifier: ') 
result = train_grid(trainx_all, trainy, model5, grid_params_model5)
find_best_sel_hyp(result)
# Feature Sets:  (3, 4, 5)
# Best hyperparameters:  {'n_neighbors': 3}
# Score:  0.827683615819209
#%% Build Featureset again
sel = (2, 4, 5)
trainx = trainx_all[sel[0]]
testx = testx_all[sel[0]]
i = 1
while(len(sel) > i):
    trainx = np.concatenate((trainx, trainx_all[sel[i]]), axis=1)
    testx = np.concatenate((testx, testx_all[sel[i]]), axis=1)
    i += 1
    
#%% XGBoost
print('XGBoost Classifier: ') 
train_grid_big_model(trainx, trainy, model2, grid_params_model2)
# Best hyperparameters
# {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}

#%% Random Forrest
print('Random Forrest Classifier: ') 
train_grid_big_model(trainx, trainy, model3, grid_params_model3)
# Best hyperparameters
# {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}

#%% evaluate model on test data
# evalutes model on test data
def eval_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_test_pred = model.predict(X_test)
    
    # Output the classification report
    report_test = classification_report(y_test, y_test_pred, target_names=['Not-FOG', 'FOG'])
    print('Test report: ')
    print(report_test)

# instantiate models
model1 = svm.SVC(C=10, gamma=0.001, kernel='rbf') # SVM
model2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=300, objective='binary:logistic')
model3 = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=500, random_state=42)
model4 = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=1000)
model5 = KNeighborsClassifier(n_neighbors=3)

# naive bayes classifiers
gnb_classifier = GaussianNB()

# ensemble model
model_svm = svm.SVC(C=10, gamma=0.001, kernel='rbf')
model_rf = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=500, random_state=42)
model_lr = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=1000)
model_knn = KNeighborsClassifier(n_neighbors=3)

model_ensemble = VotingClassifier(estimators=[('svm', model_svm), 
                                              ('random_forest', model_rf),  
                                              ('knn', model_knn)], 
                                  voting='hard')



# evaluate models
print('Gaussian Naive Bayes: ')
eval_model(gnb_classifier, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('SVM: ')
eval_model(model1, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('XGBoost: ')
eval_model(model2, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('Random Forrest: ')
eval_model(model3, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('Logistic Regression: ')
eval_model(model4, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('KNN: ')
eval_model(model5, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)
print('ensemble model: ')
eval_model(model_ensemble, X_train=trainx, y_train=trainy, X_test=testx, y_test=testy)

#%% set size vs. performance graph
# training ratios to be checked
train_sizes = np.arange(0.10, 1, 0.01)
# store accuracy, F1-Micro, F1-Macro values
hist_acc = np.zeros(len(train_sizes))
hist_f1_micro = np.zeros(len(train_sizes))
hist_f1_macro = np.zeros(len(train_sizes))
# number of train samples
n_train = trainx.shape[0]
for i in range(len(train_sizes)):
    # number of train samples
    n_sample = int(train_sizes[i] * n_train)
    # create subsample of dataset
    trainx_sub = trainx[:n_sample, :]
    trainy_sub = trainy[:n_sample]
    # create the best model 
    model = svm.SVC(C=10, gamma=0.001, kernel='rbf')
    # train model
    model.fit(trainx_sub, trainy_sub)
    # predict on test data
    testy_predict = model.predict(testx)
    # calculate accuracy
    hist_acc[i] = accuracy_score(testy, testy_predict)
    hist_f1_micro[i] = f1_score(testy, testy_predict, average='micro')
    hist_f1_macro[i] = f1_score(testy, testy_predict, average='macro')
# plot performance graph
plt.figure()
plt.scatter(train_sizes, hist_acc, color='g', alpha=0.1, s = 10)
# plt.xlabel('size of training (%)')
# plt.ylabel('test accuracy (%)')
# plt.title('set size vs. performance graph')
# plt.show()

# plt.figure()
# plt.scatter(train_sizes, hist_f1_micro, color='orange', alpha=0.1, s=10)
# plt.xlabel('size of training (%)')
# plt.ylabel('test F1-micr')
# plt.title('set size vs. performance graph')
# plt.show()

# plt.figure()
plt.scatter(train_sizes, hist_f1_macro, color='orange', alpha=0.1, s=10)
# plt.xlabel('size of training (%)')
# plt.ylabel('test F1-macro')
# plt.title('set size vs. performance graph')
# plt.show()

#%
# degree of polynomial
d = 4
coeff_acc = np.polyfit(train_sizes, hist_acc, d)
p_acc = np.poly1d(coeff_acc)
acc_smooth = p_acc(train_sizes)

coeff_f1_micro = np.polyfit(train_sizes, hist_f1_micro, d)
p_f1_micro = np.poly1d(coeff_f1_micro)
f1_micro_smooth = p_f1_micro(train_sizes)

coeff_f1_macro = np.polyfit(train_sizes, hist_f1_macro, d)
p_f1_macro = np.poly1d(coeff_f1_macro)
f1_macro_smooth = p_f1_macro(train_sizes)

# plt.figure()
plt.plot(train_sizes, acc_smooth, label='accuracy', color='g')
# plt.plot(train_sizes, f1_micro_smooth, label='F1-micro', color='orange')
plt.plot(train_sizes, f1_macro_smooth, label='F1-macro', color='orange')
plt.xlabel('size of training (%)')
plt.ylabel('test performance')
plt.title('set size vs. performance graph')
plt.grid()
plt.legend()
plt.show()
#%% ablation study
# best feature selection
sel = [2, 4, 5]
feature_names = ['n-gram count vector', 'cluster count vector', 'n-gram tf-idf', 'cluster tf-idf', 'word embedding word2vec', 'word embedding glove']
for i in range(len(sel)):
    # ablation feature selection: remove one feature at a time
    sel_abl = sel.copy()
    print(feature_names[sel[i]] + ' removed!')
    sel_abl.pop(i)
    trainx_abl = trainx_all[sel_abl[0]]
    testx_abl = testx_all[sel_abl[0]]
    j = 1
    while(len(sel_abl) > j):
        trainx_abl = np.concatenate((trainx_abl, trainx_all[sel_abl[j]]), axis=1)
        testx_abl = np.concatenate((testx_abl, testx_all[sel_abl[j]]), axis=1)
        j += 1
    
    # create the best model and evluate
    model = svm.SVC(C=10, gamma=0.001, kernel='rbf')
    eval_model(model, trainx_abl, trainy, testx_abl, testy)
    
    


















