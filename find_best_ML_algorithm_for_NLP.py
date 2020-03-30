#!/usr/bin/python
#%%
import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import os
path = os.path.join(os.path.dirname(__file__), '../ML/tools/')
#%%
### define method for selecting best algorithm for text classification
def NLPML(text = "E:/Google Drive/git/Utilities/word_data_fixed.pkl",
            lable = "E:/Google Drive/git/Utilities/email_authors_fixed.pkl",
            newText = ['I like to go out what about you?','I need some money plaese let me know if you can borrow me','love home and shopping love home and shopping love home and shopping','baseball work football','I am going to buy a new dress and makeup and lipstick','counterparti for ena nontermin inthemoney posit base upon fmtm inform']):
    words_file = text
    authors_file= lable

    ### reading files
    authors_file_handler = open(authors_file, "rb")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    ### ten persent of the data is considered as test data
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)


    ### feature selection
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    features_train_transformed_1 = features_train_transformed[:int(len(features_train_transformed)/100)]
    labels_train_1 = labels_train[:int(len(labels_train)/100)]
    #return features_train_transformed, features_test_transformed, labels_train, labels_test
    results = []

    svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1,10],
                     'C': [0.1,1, 10, 100, 1000, 6000]},
                    {'kernel': ['linear', 'sigmoid','poly'], 'C': [1, 10, 100, 1000,6000]}]

    svm_clf = GridSearchCV(svm.SVC(), svm_tuned_parameters)
    pred = svm_clf.fit(features_train_transformed_1, labels_train_1).predict(features_test_transformed)
    svm_accuracy = metrics.accuracy_score(labels_test, pred)
    results.append({'algorithm':'SVM', 'Accuracy':svm_accuracy})
    print("SVM Accuracy:",metrics.accuracy_score(labels_test, pred))
    print('SVM Best parameters: ',svm_clf.best_params_)
    print()

    nb_tuned_parameters = [{'var_smoothing':[1e-10,1e-9,1e-5,1e-3,1,10]}]
    nb_clf = GridSearchCV(GaussianNB(),nb_tuned_parameters)
    pred = nb_clf.fit(features_train_transformed_1, labels_train_1).predict(features_test_transformed)
    nb_accuracy = metrics.accuracy_score(labels_test, pred)
    print("GaussianNB Accuracy:",nb_accuracy)
    print('NB Best parameters: ',nb_clf.best_params_)
    results.append({'algorithm':GaussianNB(), 'Accuracy':nb_accuracy})
    print()

    dt_clf = tree.DecisionTreeClassifier()
    pred = dt_clf.fit(features_train_transformed_1, labels_train_1).predict(features_test_transformed)
    dt_accuracy = metrics.accuracy_score(labels_test, pred)
    print("DecisionTree Accuracy:",dt_accuracy)
    results.append({'algorithm': tree.DecisionTreeClassifier(), 'Accuracy':dt_accuracy})
    print()

    rf_tuned_parameters = [{'n_estimators':[10,25,50,75,100,150]}]
    rf_clf = GridSearchCV(RandomForestClassifier(),rf_tuned_parameters)
    pred = rf_clf.fit(features_train_transformed_1, labels_train_1).predict(features_test_transformed)
    dt_accuracy = metrics.accuracy_score(labels_test, pred)
    print("RandomForest Accuracy:",dt_accuracy)
    print('RandomForest Best parameters: ',rf_clf.best_params_)
    results.append({'algorithm':RandomForestClassifier(), 'Accuracy':dt_accuracy})
    print()

    
"""     best = max(results, key=lambda x: x['Accuracy'])
    transformed_text = vectorizer.transform(newText)
    clf = best['algorithm']
    pred_text = clf.fit(features_train_transformed, labels_train).predict(transformed_text)
    print(pred_text) """
# %%
NLPML()



# %%
