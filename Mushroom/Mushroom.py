import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus


# Read mushroom data

mushroom_df = pd.read_csv("mushrooms.csv")

# Convert data frame to categorical variables
def get_features(data):
    features = [f for f in data]
    return features

def make_categories(data):
    features = get_features(data)
    for feature in features:
        data[feature] = data[feature].astype("category")
        data[feature] = data[feature].cat.codes
    return data

mushroom = make_categories(mushroom_df)

# Check to see if any features have only 1 value. Drop those features
features = get_features(mushroom)
empty_dict = {}
for feature in features:
    empty_dict[feature] = mushroom[feature].nunique()
print(empty_dict)


mushroom.drop("veil-type", axis=1, inplace=True)


# Create a correlation heatmap
print(sns.heatmap(mushroom.corr()))

# Sort correlations
mushroom_corr = mushroom.corr()
print(mushroom_corr.sort_values(["class"], ascending=False))

# Separate features and target
X = mushroom.drop("class", axis=1)
y = mushroom["class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Random Forest Model
forest = RandomForestClassifier()
forest_params = {"n_estimators": [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000],
                "criterion": ["gini", "entropy"],
                "max_features": ["auto", "log2"]}
forest_grid = GridSearchCV(forest, forest_params, cv=10)
forest_grid.fit(X_train, y_train)

forest_grid.best_params_
forest_grid.best_score_
y_pred = forest_grid.predict(X_test)
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

# GridSearchCV Logistic Regression Model
logistic = LogisticRegression()
logistic_params = {"penalty": ["l1", "l2"],
                "C": [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000]}
logistic_grid = GridSearchCV(logistic, logistic_params, cv=10)
logistic_grid.fit(X_train, y_train)


logistic_grid.best_params_
logistic_grid.best_score_

# Non GridSearchCV Logistic Regression (for coefficient analysis)
logistic2 = LogisticRegression(penalty="l1", C=11)
logistic2.fit(X_train, y_train)
y_pred = logistic2.predict(X_test)

# Sort most predictive coefficients
X_features = get_features(X)
coefficients = logistic2.coef_.transpose()
coeff_df = pd.DataFrame(coefficients, index=X_features, columns= ["importance"])
coeff_df.sort_values("importance")

print(classification_report(y_pred, y_test))


# Logistic Regression with SelectKBest Pipeline
five_best = SelectKBest(k=5)
top_five_log = Pipeline(steps=[("five_best", five_best), ("logistic2", logistic2)])
top_five_log.fit(X_train, y_train)
y_pred2 = top_five_log.predict(X_test)
print(classification_report(y_pred2, y_test))
print(confusion_matrix(y_pred2, y_test))

## Decision tree
d_tree = tree.DecisionTreeClassifier(criterion="gini")
d_tree.fit(X_train, y_train)
y_tree_pred = d_tree.predict(X_test)
print(classification_report(y_tree_pred, y_test))
dot_data = StringIO()
dot_data = tree.export_graphviz(d_tree, out_file=None,feature_names=X_features)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
