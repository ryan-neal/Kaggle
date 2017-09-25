
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.stats import norm, skew, boxcox
get_ipython().magic('matplotlib inline')
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile


# ## Frame the Problem
# 1. What is the objective?<br>
# The objective is to predict housing prices<br>
# 
# 2. How will the solution be used?<br>
# A good business application would be to compare predicted house values using the model to listed house prices at realtor websites. Then, with proper financing, one could buy the houses with the largest "margins of safety" and sell them for a profit<br>
# 
# 3. What methods are used?<br>
# Linear regression with Ridge, Lasso, and Elastic versions<br>
# 
# 4. How is performance measured?<br>
# RMSE is used to measure the performance of the model<br>

# ## Get Data

# In[2]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# ## Explore Data

# In[3]:


df_train.head()


# There is an extra Id column. We will drop this.

# In[4]:


df_train.drop("Id", axis=1, inplace=True)


# In[5]:


df_train.describe()


# In[6]:


df_train.info()


# Alley, PoolQC, Fence, MiscFeature, FireplaceQu, Garage attributes, Basement Attributes, and MasVnrType all have missing values.

# In[7]:


corr = df_train.corr()
corr.sort_values(["SalePrice"], ascending=False, inplace=True)
print(corr.SalePrice)


# In[8]:


sns.distplot(df_train["GrLivArea"])


# In[9]:


extreme_area = df_train[df_train["GrLivArea"] > 4000]


# In[10]:


sns.jointplot(data=df_train, x="GrLivArea", y="SalePrice")


# There seems to be outliers after 4000 area. We will remove these.

# In[11]:


df_train = df_train[df_train["GrLivArea"] < 4000]


# In[12]:


sns.jointplot(data=df_train, x="GrLivArea", y="SalePrice")


# We still might have some issues with large living areas, but let's leave it for now.

# In[13]:


def create_box_chart(independent, dependent="SalePrice"):
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=independent,y=dependent,data=df_train,palette='rainbow')


# In[14]:


create_box_chart("OverallQual")


# In[15]:


sns.countplot(df_train["GarageCars"])


# In[16]:


create_box_chart("GarageCars")


# It looks like 4 car garages actually sell for less than 3 car garages.

# In[17]:


df_train[df_train["GarageCars"]==4]


# In[18]:


garage_variables = [f for f in df_train.columns if "Garage" in f]
garage_variables


# In[19]:


df_train[garage_variables].isnull().any()


# In[20]:


sns.jointplot(data=df_train, x="GarageYrBlt", y="SalePrice")


# In[21]:


sns.jointplot(data=df_train, x="GarageYrBlt", y="SalePrice")


# In[22]:


create_box_chart("GarageQual")


# ## Prepare Data

# In[23]:


# clean garage variables NA means no garage
df_train["GarageType"].fillna("none", inplace=True)
df_train["GarageQual"].fillna("none", inplace=True)
df_train["GarageCond"].fillna("none", inplace=True)
df_train["GarageFinish"].fillna("none", inplace=True)


# In[24]:


df_train["YearBuilt"].corr(df_train["GarageYrBlt"])


# In[25]:


garage_year_built = df_train["GarageYrBlt"]


# In[26]:


df_train.drop("GarageYrBlt", axis=1, inplace=True)


# In[27]:


corr = df_train.corr()
corr.sort_values(["SalePrice"], ascending=False, inplace=True)
print(corr.SalePrice)


# In[28]:


quantitative = [f for f in df_train.columns if df_train.dtypes[f] != object]
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == object]


# In[29]:


# MasVnrArea and LotFrontage still have nans. In both cases nan probably means 0
df_train["LotFrontage"].fillna(0, inplace=True)
df_train["MasVnrArea"].fillna(0, inplace=True)


# In[30]:


df_train[qualitative].info()


# A lot of the qualitative features have missing values

# In[31]:


# nan Alley likely means none
df_train["Alley"].fillna("none", inplace=True)
df_train["MasVnrType"].fillna("None", inplace=True)


# In[32]:


create_box_chart("MasVnrType")


# In[33]:


## Get basement variables
basement_variables = [f for f in df_train.columns if "Bsmt" in f]


# In[34]:


# fillnas
df_train['BsmtQual'].fillna(0, inplace=True)
df_train['BsmtCond'].fillna(0, inplace=True)
df_train['BsmtExposure'].fillna(0, inplace=True)
df_train['BsmtFinType1'].fillna(0, inplace=True)
df_train['BsmtFinType2'].fillna(0, inplace=True)


# In[35]:


# fill fireplace, pool, fence and misc with 0
df_train['FireplaceQu'].fillna(0, inplace=True)
df_train['PoolQC'].fillna(0, inplace=True)
df_train['Fence'].fillna(0, inplace=True)
df_train['MiscFeature'].fillna(0, inplace=True)


# In[36]:


df_train["FireplaceQu"] = df_train["FireplaceQu"].astype("category")
df_train["FireplaceQu_Cat"] = df_train["FireplaceQu"].cat.codes


# In[37]:


for col in df_train[qualitative].columns:
    df_train[col] = df_train[col].astype("category")
    df_train[col] = df_train[col].cat.codes


# In[38]:


corr = df_train.corr()
corr.sort_values(["SalePrice"], ascending=False, inplace=True)
print(corr.SalePrice)


# In[39]:


sns.countplot(df_train["FireplaceQu_Cat"])


# In[40]:


df_train[quantitative].info()


# In[41]:


total_square_feet = [f for f in df_train[quantitative] if "SF" in f]


# In[42]:


total_square_feet


# In[43]:


df_train["SquareFeet"] = df_train["TotalBsmtSF"] + df_train["1stFlrSF"] + df_train["2ndFlrSF"]


# ## First model
# - skewed data
# - no interaction terms

# In[44]:


reg = LinearRegression()
X = df_train.drop("SalePrice", axis=1)
y = np.log1p(df_train["SalePrice"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)


# In[104]:


X["SalePrice"]


# In[46]:


def scatter_results(test, predicted):
    plt.scatter(test, predicted)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')


# In[47]:


scatter_results(y_test, y_pred)


# In[48]:


def errors(test, predicted):
    print('MAE:', metrics.mean_absolute_error(test, predicted))
    print('MSE:', metrics.mean_squared_error(test, predicted))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(test, predicted)))


# In[49]:


errors(y_test, y_pred)


# The simple model does surprisingly well

# ## Ridge

# In[50]:


ridge = RidgeCV(alphas=[.01, .03, .06, .1, .3, .6, 1, 3, 6, 10, 30, 60])


# In[51]:


ridge.fit(X_train, y_train)
y_pred_rid = ridge.predict(X_test)
scatter_results(y_test, y_pred_rid)


# In[52]:


errors(y_test, y_pred_rid)


# In[53]:


alpha = ridge.alpha_


# In[54]:


ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)

ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)


# In[55]:


y_pred_rdg = ridge.predict(X_test)

scatter_results(y_test, y_pred_rdg)


# In[56]:


errors(y_test, y_pred_rdg)


# In[57]:


coefs = pd.Series(ridge.coef_, index = X_train.columns)

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")


# ## Lasso

# In[58]:


lasso = LassoCV(alphas=[.01, .03, .06, .1, .3, .6, 1, 3, 6, 10, 30, 60], max_iter=50000, cv=10)


# In[59]:


lasso.fit(X_train, y_train)


# In[60]:


y_pred_las = lasso.predict(X_test)


# In[61]:


scatter_results(y_test, y_pred)


# In[62]:


alpha = lasso.alpha_
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)


# In[63]:


y_pred_las2 = lasso.predict(X_test)
plt.scatter(y_test, y_pred_las2)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[64]:


errors(y_test, y_pred_las2)


# ## Elastic

# In[65]:


elastic = ElasticNetCV(alphas = [.01, .03, .06, .1, .3, .6, 1, 3, 6, 10, 30, 60], max_iter= 50000, cv=10,
                       l1_ratio=[.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99])


# In[66]:


elastic.fit(X_train, y_train)


# In[67]:


y_pred_elastic = elastic.predict(X_test)


# In[68]:


scatter_results(y_test, y_pred_elastic)


# In[69]:


errors(y_test, y_pred_elastic)


# Ridge performed the best. Let's see if fixing skewness or adding polynomial terms improves anything

# ## Fixing Skewness

# In[70]:


sns.distplot(y)


# In[71]:


skewness = [[f, skew(df_train[f])] for f in df_train[quantitative].columns]


# In[72]:


skewness = pd.DataFrame(skewness, columns = ["Feature", "skew"])


# In[73]:


skewness.set_index("Feature", inplace=True)


# In[74]:


skewness.sort_values("skew", ascending=False)


# In[75]:


df_train2 = df_train.copy()


# In[76]:


for feature in skewness.index:
    if abs(skew(df_train[feature])) > .75:
        df_train2[feature] = np.log1p(df_train[feature])


# In[77]:


y = df_train2["SalePrice"]
X = df_train2.drop("SalePrice", axis=1)


# In[78]:


regr = LinearRegression()


# In[79]:


X_train, X_test, y_train, y_split = train_test_split(X, y, test_size = 0.3)


# In[80]:


regr.fit(X_train, y_train)


# In[81]:


y_pred_norm = regr.predict(X_test)


# In[82]:


scatter_results(y_test, y_pred_norm)


# In[83]:


errors(y_test, y_pred_norm)


# Ya that did not really work

# ## Adding quadratic terms

# In[84]:


poly = PolynomialFeatures(degree=2, include_bias = False)


# In[85]:


X = df_train.drop("SalePrice", axis=1)
y = np.log1p(df_train["SalePrice"])


# In[86]:


X_train, X_test, y_train, y_split = train_test_split(X, y, test_size = 0.3)


# In[87]:


the_X = poly.fit_transform(X)


# In[88]:


re = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(re,parameters, cv=10)
grid.fit(the_X, y)


# In[89]:


y_pred_quad = grid.predict(poly.fit_transform(X_test))


# In[90]:


errors(y_test,  y_pred_quad)


# In[105]:


coefs = pd.Series(grid.best_estimator_.coef_, poly.get_feature_names())

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Quadratic Model")


# In[106]:


coefs


# In[92]:


all_columns = [f for f in df_train.columns]


# In[93]:


df_train["Pool * BsmtHalfBath"] = df_train[all_columns[47]] * df_train[all_columns[70]]
#df_train["BsmtHalfBathSquared"] = df_train[all_columns[47]] **2
df_train["BsmtBaths"] = df_train[all_columns[46]] * df_train[all_columns[47]]
df_train["Slope * BsmtHalfBath"] = df_train[all_columns[10]] * df_train[all_columns[47]]
#df_train["Slope * BldgType"] = df_train[all_columns[10]] * df_train[all_columns[14]]
df_train["Slope * Paved"] = df_train[all_columns[10]] * df_train[all_columns[63]]
df_train["CountourSquared"] = df_train["LandContour"] **2
df_train["FirePlacesSquared"] = df_train[all_columns[55]]
df_train["NotBsmtHalfBath"] = df_train["HalfBath"] - df_train["BsmtHalfBath"]
df_train["Qual * Cond"] = df_train["OverallQual"] * df_train["OverallCond"]


# In[94]:


reg = LinearRegression()
X = df_train.drop("SalePrice", axis=1)
y = np.log1p(df_train["SalePrice"])
X_train, X_test, y_train, y_split = train_test_split(X, y, test_size = 0.3)
reg.fit(X_train, y_train)
y_pred_new = reg.predict(X_test)


# In[95]:


scatter_results(y_test, y_pred_new)


# In[96]:


errors(y_test, y_pred_new)


# In[97]:


sns.distplot((y_test-y_pred_new),bins=50)


# ## Return to the Ridge

# In[98]:


ridge = RidgeCV(alphas=[.01, .03, .06, .1, .3, .6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y_train)
y_pred_rid = ridge.predict(X_test)
scatter_results(y_test, y_pred_rid)


# In[99]:


errors(y_test, y_pred_rid)


# In[100]:


ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)

ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)


# In[101]:


y_pred_rdg = ridge.predict(X_test)

scatter_results(y_test, y_pred_rdg)


# In[102]:


errors(y_test, y_pred_rdg)


# In[103]:


coefs = pd.Series(ridge.coef_, index = X_train.columns)

imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")


# In[ ]:




