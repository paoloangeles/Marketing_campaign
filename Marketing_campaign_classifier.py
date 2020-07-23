#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: paoloangeles
"""

## IMPORT REQUIRED MODULES ##
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, ensemble, linear_model, model_selection, feature_selection, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


## READ DATASET ##
dataset = pd.read_excel("dataset.xlsx")


## DATA PREPROCESSING ##
## Downcast dataframe dataset in order to save memory
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df

dataset = downcast(dataset)
dataset = dataset.drop(columns = "client", axis = 'columns')


## DATA CLEANING ##
## Check for mising values
print(dataset.apply(lambda x: sum(x.isnull()), axis=0))

## Split data features into numeric and category types
x_old = dataset.drop(columns = "subscribed", axis = 'columns')
x_old_category = x_old[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]]
x_old_numeric = x_old.drop(columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"])

## Use one hot encoding for category features as sklearn library gives important to number amounts
encoder = preprocessing.OneHotEncoder()
x_category = encoder.fit_transform(x_old_category)
x_category = pd.DataFrame(x_category.toarray())
x_category.columns = encoder.get_feature_names()
x = pd.concat([x_old_numeric, x_category], axis = 1)

## Use label encoding on output variable, 0 being not subscribed and 1 being subscribed
y_old = dataset["subscribed"]
y = preprocessing.LabelEncoder().fit_transform(y_old)


## MODEL RUN WITH ALL FEATURES/VARIABLES ##
## Split data into training and testing set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, shuffle = True, stratify = y)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

## Logistic model
logit_model = linear_model.LogisticRegression(max_iter = 5000)
scores = model_selection.cross_val_score(logit_model, x_train, y_train, cv = 10)
print("Cross-validated logistic model scores:", scores)

## Fit logistic model to training data
logit_model.fit(x_train, y_train)

## Evaluate model on test data
print("Logistic model test score:", logit_model.score(x_test, y_test))
y_pred0 = logit_model.predict(x_test)
metric_score_logit_model = metrics.classification_report(y_test, y_pred0)


## Evaluate model 1 on training data
svc_model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
scores = model_selection.cross_val_score(svc_model, x_train, y_train, cv = 10)
print("Cross-validated SVM model scores:", scores)

## Fit model to training data
svc_model.fit(x_train, y_train)

## Evaluate model on test data
print("SVM model test score:", svc_model.score(x_test, y_test))
y_pred = svc_model.predict(x_test)
metric_score_svm_model = metrics.classification_report(y_test, y_pred)

## Evaluate model 2 on training data
rf_model = ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
scores2 = model_selection.cross_val_score(rf_model, x_train, y_train, cv = 10)
print("Cross-validated RF model scores:", scores2)

## Fit model to training data
rf_model.fit(x_train, y_train)

## Evaluate model on test data
print("RF model test score:", rf_model.score(x_test, y_test))
y_pred2 = rf_model.predict(x_test)
y_prob2 = probs = pd.DataFrame(rf_model.predict_proba(x_test))
metric_score_rf_model = metrics.classification_report(y_test, y_pred2)

rf_accuracy = metrics.accuracy_score(y_test, y_pred2)     
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])       
rf_confus_matrix = metrics.confusion_matrix(y_test, y_pred2) 
rf_classification_report = metrics.classification_report(y_test, y_pred2)
rf_precision = metrics.precision_score(y_test, y_pred2, pos_label=1)
rf_recall = metrics.recall_score(y_test, y_pred2, pos_label=1)
rf_f1 = metrics.f1_score(y_test, y_pred2, pos_label=1)


## FEATURE EVALUATION AND SELECTION ##
selector = feature_selection.SelectKBest(feature_selection.f_classif, k=5)
x_kbest = selector.fit_transform(x_train, y_train)
x_kbest_test = selector.fit_transform(x_test, y_test)
selector.fit(x_train, y_train)

## Print ANOVA F-Scores for the features
for i in range(len(selector.scores_)):
	print(str(x.columns.values[i] + " :" + str(selector.scores_[i])))
    
## Plot the ANOVA F-Scores
plt.bar(x.columns.values, selector.scores_)
plt.show()

## Plot top 3 features
plt.figure()
plt.bar([x.columns.values[1], x.columns.values[9], x.columns.values[3]], np.array([selector.scores_[1], selector.scores_[9], selector.scores_[3]]))
plt.xlabel("Feature/variable")
plt.ylabel("ANOVA F-Score")
plt.show()


## FIND HIGHLY CORRELATED VARIABLES ##
## Label encode variables for correlation measures - do not require to one hot encode for correlation
label_encoder = preprocessing.LabelEncoder()
x_category_label = x_old_category.apply(label_encoder.fit_transform)
x_label = pd.concat([x_old_numeric, x_category_label], axis = 1)

## Plot correlation matrix which is based on Pearson's correlation coefficient
plt.figure(figsize=(12,10))
cor = x_label.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


## MODEL RUN USING ONLY 5 FEATURES ##
## Logistic model run
logit_model_kbest = linear_model.LogisticRegression(max_iter = 5000)
scores = model_selection.cross_val_score(logit_model_kbest, x_kbest, y_train, cv = 10)
print("Cross-validated scores:", scores)

## Fit model to training data
logit_model_kbest.fit(x_kbest, y_train)

## Evaluate model on test data
print("Score:", logit_model_kbest.score(x_kbest_test, y_test))
y_pred3 = logit_model_kbest.predict(x_kbest_test)
metric_score_logit_model_kbest = metrics.classification_report(y_test, y_pred3)


## SVM model run
svm_model_kbest = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
scores3 = model_selection.cross_val_score(svm_model_kbest, x_kbest, y_train, cv = 10)
print("Cross-validated scores:", scores3)

## Fit model to training data
svm_model_kbest.fit(x_kbest, y_train)

## Evaluate model on test data
print("Score:", svm_model_kbest.score(x_kbest_test, y_test))
y_pred4 = svm_model_kbest.predict(x_kbest_test)
metric_score_svm_model_kbest = metrics.classification_report(y_test, y_pred4)

## RF model run
rf_model_kbest = ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
scores4 = model_selection.cross_val_score(rf_model_kbest, x_kbest, y_train, cv = 10)
print("Cross-validated scores:", scores4)

## Fit model to training data
rf_model_kbest.fit(x_kbest, y_train)

## Evaluate model on test data
print("Score:", rf_model_kbest.score(x_kbest_test, y_test))
y_pred5 = rf_model_kbest.predict(x_kbest_test)
metric_score_rf_model_kbest = metrics.classification_report(y_test, y_pred5)


## MODEL SAVING/LOADING ##
# # save the models
# filename = 'logit_model_model.sav'
# pickle.dump(model, open(filename, 'wb'))
 
 
# # load the models
# logit_model = pickle.load(open('logit_model_file.sav', 'rb'))
# svc_model = pickle.load(open('svc_model_file.sav', 'rb'))
# rf_model = pickle.load(open('rf_model_file.sav', 'rb'))
# logit_model_kbest = pickle.load(open('logit_model_kbest_file.sav', 'rb'))
# svc_model_kbest = pickle.load(open('clf3_file.sav', 'rb'))
# rf_model_kbest = pickle.load(open('clf4_file.sav', 'rb'))
# selector = pickle.load(open('selector_file.sav', 'rb'))