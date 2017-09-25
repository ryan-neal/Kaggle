
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


# In[59]:


mushroom_df = pd.read_csv("mushrooms.csv")


# In[3]:


def get_features(data):
    features = [f for f in data]
    return features


# In[4]:


def make_categories(data):
    features = get_features(data)
    for feature in features:
        data[feature] = data[feature].astype("category")
        data[feature] = data[feature].cat.codes
    return data


# In[70]:


mushroom = make_categories(mushroom_df)


# In[77]:


features = get_features(mushroom)
empty_dict = {}
for feature in features:
    empty_dict[feature] = mushroom[feature].nunique()
print(empty_dict)


# In[78]:


mushroom.drop("veil-type", axis=1, inplace=True)


# In[6]:


sns.heatmap(mushroom.corr())


# In[7]:


mushroom_corr = mushroom.corr()


# In[8]:


print(mushroom_corr.sort_values(["class"], ascending=False))


# In[82]:


X = mushroom.drop("class", axis=1)
y = mushroom["class"]


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Random Forest

# In[12]:


forest = RandomForestClassifier()
forest_params = {"n_estimators": [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000],
                "criterion": ["gini", "entropy"],
                "max_features": ["auto", "log2"]}
forest_grid = GridSearchCV(forest, forest_params, cv=10)
forest_grid.fit(X_train, y_train)


# In[14]:


forest_grid.best_params_


# In[15]:


forest_grid.best_score_


# In[16]:


y_pred = forest_grid.predict(X_test)


# In[20]:


print(classification_report(y_pred, y_test))


# In[22]:


print(confusion_matrix(y_pred, y_test))


# ## Logistic Regression

# In[24]:


logistic = LogisticRegression()
logistic_params = {"penalty": ["l1", "l2"],
                "C": [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000]}
logistic_grid = GridSearchCV(logistic, logistic_params, cv=10)
logistic_grid.fit(X_train, y_train)


# In[25]:


logistic_grid.best_params_


# In[27]:


logistic_grid.best_score_


# In[57]:


logistic2 = LogisticRegression(penalty="l1", C=11)
logistic2.fit(X_train, y_train)
y_pred = logistic2.predict(X_test)
X_features = get_features(X)
coefficients = logistic2.coef_.transpose()
coeff_df = pd.DataFrame(coefficients, index=X_features, columns= ["importance"])
coeff_df.sort_values("importance")


# In[40]:


print(classification_report(y_pred, y_test))


# ## Logistic Regression with SelectKBest Pipeline

# In[84]:


five_best = SelectKBest(k=5)
top_five_log = Pipeline(steps=[("five_best", five_best), ("logistic2", logistic2)])


# In[85]:


top_five_log.fit(X_train, y_train)


# In[86]:


y_pred2 = top_five_log.predict(X_test)


# In[87]:


y_pred2


# In[88]:


print(classification_report(y_pred2, y_test))


# In[89]:


print(confusion_matrix(y_pred2, y_test))


# In[ ]:
