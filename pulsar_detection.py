#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix ,f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import plotly.express as px


# In[2]:


data = pd.read_csv("Data/HTRU_2.csv")
x = data.iloc[:, 0:7]
y = data.iloc[:, -1]


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[6]:


standard_scale = StandardScaler()
x_train_scaled = standard_scale.fit_transform(x_train)


# In[9]:


smt = SMOTE(sampling_strategy=0.5, random_state=42)
x_train_sm, y_train_sm = smt.fit_sample(x_train_scaled, y_train)


# # Random Forest Classifier

# In[11]:


random_forest=RandomForestClassifier()
random_forest.fit(x_train_sm, y_train_sm)


# In[12]:


random_forest.feature_importances_


# In[17]:


y_train_pred_random_forest=cross_val_predict(random_forest, x_train_sm, y_train_sm, cv=5)


# In[18]:


confusion_matrix_random_forest = confusion_matrix(y_train_sm, y_train_pred_random_forest)

confusion_matrix_random_forest


# In[19]:


# precision, accuracy and recall for random forest classifier

print(accuracy_score(y_train_sm, y_train_pred_random_forest))
print(precision_score(y_train_sm, y_train_pred_random_forest))
print(recall_score(y_train_sm, y_train_pred_random_forest))
print(f1_score(y_train_sm, y_train_pred_random_forest))


# In[20]:


y_test_pred_random_forest=random_forest.predict(standard_scale.fit_transform(x_test))


# In[21]:


# precision, accuracy and recall for random forest classifier

print(accuracy_score(y_test, y_test_pred_random_forest))
print(precision_score(y_test, y_test_pred_random_forest))
print(recall_score(y_test, y_test_pred_random_forest))
print(f1_score(y_test, y_test_pred_random_forest))


# # Support Vector Machines

# In[24]:


gaussian_svm = SVC(kernel="poly",degree=3, C=5, probability=True)
gaussian_svm.fit(x_train_sm, y_train_sm)


# In[25]:


y_train_pred_svm=cross_val_predict(gaussian_svm, x_train_sm, y_train_sm, cv=5)


# In[26]:


print(accuracy_score(y_train_sm, y_train_pred_svm))
print(precision_score(y_train_sm, y_train_pred_svm))
print(recall_score(y_train_sm, y_train_pred_svm))
print(f1_score(y_train_sm, y_train_pred_svm))


# In[27]:


y_gaussian_svm=gaussian_svm.predict(standard_scale.fit_transform(x_test))

print(accuracy_score(y_test, y_gaussian_svm))
print(precision_score(y_test, y_gaussian_svm))
print(recall_score(y_test, y_gaussian_svm))
print(f1_score(y_test, y_gaussian_svm))


# # Logistic Regression

# In[34]:


logistic_regression=LogisticRegression()
logistic_regression.fit(x_train_sm, y_train_sm)


# In[35]:


y_train_pred_log=cross_val_predict(logistic_regression, x_train_sm, y_train_sm, cv=5)


# In[36]:


# log confusion matrix

confusion_matrix_log = confusion_matrix(y_train_sm, y_train_pred_log)

confusion_matrix_log


# In[37]:


# precision, accuracy and recall for log classifier

print(accuracy_score(y_train_sm, y_train_pred_log))
print(precision_score(y_train_sm, y_train_pred_log))
print(recall_score(y_train_sm, y_train_pred_log))
print(f1_score(y_train_sm, y_train_pred_log))


# In[38]:


y_logistic_regression=logistic_regression.predict(standard_scale.fit_transform(x_test))

print(accuracy_score(y_test, y_logistic_regression))
print(precision_score(y_test, y_logistic_regression))
print(recall_score(y_test, y_logistic_regression))
print(f1_score(y_test, y_logistic_regression))


# # Adaboost

# In[40]:


adaboost_classifier=AdaBoostClassifier(
    RandomForestClassifier(), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5
)

adaboost_classifier.fit(x_train_sm, y_train_sm)


# In[41]:


# adaboost metrics

y_train_pred_adaboost=cross_val_predict(adaboost_classifier, x_train_sm, y_train_sm, cv=5)


# In[42]:


# adaboost confusion matrix

confusion_matrix_adaboost = confusion_matrix(y_train_sm, y_train_pred_adaboost)

confusion_matrix_adaboost


# In[43]:


# precision, accuracy and recall for adaboost classifier

print(accuracy_score(y_train_sm, y_train_pred_adaboost))
print(precision_score(y_train_sm, y_train_pred_adaboost))
print(recall_score(y_train_sm, y_train_pred_adaboost))
print(f1_score(y_train_sm, y_train_pred_adaboost))


# In[44]:


y_adaboost_classifier=adaboost_classifier.predict(standard_scale.fit_transform(x_test))

print(accuracy_score(y_test, y_adaboost_classifier))
print(precision_score(y_test, y_adaboost_classifier))
print(recall_score(y_test, y_adaboost_classifier))
print(f1_score(y_test, y_adaboost_classifier))


# # SGDC

# In[46]:


sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(x_train_sm, y_train_sm)


# In[47]:


y_train_pred_sgd_clf=cross_val_predict(sgd_clf, x_train_sm, y_train_sm, cv=5)


# In[48]:


# sgd_clf confusion matrix

confusion_matrix_sgd_clf = confusion_matrix(y_train_sm, y_train_pred_sgd_clf)

confusion_matrix_sgd_clf


# In[49]:


# precision, accuracy and recall for sgd_clf classifier

print(accuracy_score(y_train_sm, y_train_pred_sgd_clf))
print(precision_score(y_train_sm, y_train_pred_sgd_clf))
print(recall_score(y_train_sm, y_train_pred_sgd_clf))
print(f1_score(y_train_sm, y_train_pred_sgd_clf))


# In[50]:


y_sgd_clf=sgd_clf.predict(standard_scale.fit_transform(x_test))

print(accuracy_score(y_test, y_sgd_clf))
print(precision_score(y_test, y_sgd_clf))
print(recall_score(y_test, y_sgd_clf))
print(f1_score(y_test, y_sgd_clf))


# # K Neighbors

# In[53]:


k_neig=KNeighborsClassifier(n_neighbors=15)
k_neig.fit(x_train_sm, y_train_sm)


# In[54]:


y_train_pred_k_neig=cross_val_predict(k_neig, x_train_sm, y_train_sm, cv=5)


# In[55]:


# k_neig confusion matrix

confusion_matrix_k_neig = confusion_matrix(y_train_sm, y_train_pred_k_neig)

confusion_matrix_k_neig


# In[56]:


# precision, accuracy and recall for k_neig classifier

print(accuracy_score(y_train_sm, y_train_pred_k_neig))
print(precision_score(y_train_sm, y_train_pred_k_neig))
print(recall_score(y_train_sm, y_train_pred_k_neig))
print(f1_score(y_train_sm, y_train_pred_k_neig))


# In[57]:


y_neig=k_neig.predict(standard_scale.fit_transform(x_test))

print(accuracy_score(y_test, y_neig))
print(precision_score(y_test, y_neig))
print(recall_score(y_test, y_neig))
print(f1_score(y_test, y_neig))


# # Voting Classification

# In[61]:


voting_clf=VotingClassifier(
    estimators=[('lr', logistic_regression), ('rf', random_forest), ('ab', adaboost_classifier), ('kn', k_neig), ('sm', gaussian_svm)], voting='soft'
)

voting_clf.fit(x_train_sm, y_train_sm)


# In[62]:


y_train_pred_voting_clf=cross_val_predict(voting_clf, x_train_sm, y_train_sm, cv=5)


# In[63]:


# voting_clf confusion matrix

confusion_matrix_voting_clf = confusion_matrix(y_train_sm, y_train_pred_voting_clf)

confusion_matrix_voting_clf


# In[64]:


# precision, accuracy and recall for voting_clf classifier

print(accuracy_score(y_train_sm, y_train_pred_voting_clf))
print(precision_score(y_train_sm, y_train_pred_voting_clf))
print(recall_score(y_train_sm, y_train_pred_voting_clf))
print(f1_score(y_train_sm, y_train_pred_voting_clf))


# In[65]:


y_test_pred_voting=voting_clf.predict(standard_scale.fit_transform(x_test))


# In[66]:


print(accuracy_score(y_test, y_test_pred_voting))
print(precision_score(y_test, y_test_pred_voting))
print(recall_score(y_test, y_test_pred_voting))
print(f1_score(y_test, y_test_pred_voting))


# # saving the model

# In[2]:


import pickle 

pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(adaboost_classifier, pickle_out) 
pickle_out.close()


# In[ ]:




