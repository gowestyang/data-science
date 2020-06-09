#' % Linear Regression (Python)
#' % Yang Xi
#' % 14 Nov, 2018

#%%
#+ echo = False
import warnings
warnings.filterwarnings('ignore')

#%% Load packages
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression as lm
import statsmodels.api as sm

#%% # <br>Model Fitting and Interpretation
#%% ### Independent Numeric and Categorical variables
# Simulate data
random.seed(1)
n = 9000
x1 = np.array(['A', 'B', 'C']*int(n/3))
x1 = np.random.choice(x1, n, replace=False)
x2 = np.random.uniform(0, 2, n)
x3 = np.random.uniform(0, 1, n)
e = np.random.uniform(-0.5, 0.5, n)

b1_map = {'A':-2, 'B':-1, 'C':1} # 0, 1, 3
b2 = 3

X = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3})
y = list(map(b1_map.get, x1)) + b2*x2 + e 

# Covert categorical variable
X = pd.get_dummies(X, columns=['x1'], drop_first=True) # x2, x3, x1_B, x1_C

# Fit model
lmFit = lm().fit(X, y)
print('intercept = {0:.3f}'.format(lmFit.intercept_))
print(pd.DataFrame(lmFit.coef_, index=X.columns, columns=['coef']))

#' <br>statsmodels provides more statistics
#+ wrap = False
Xc = sm.add_constant(X)
sm_fit = sm.OLS(y, Xc).fit()
print(sm_fit.summary())

#%% ### <br>Response to highly correlated numeric variables
# Simulate data
random.seed(1)
n = 10000
x1 = np.random.uniform(1, 2, n)
x2 = x1
e = np.random.uniform(-0.5, 0.5, n)

b1 = 2
b2 = 3

X = pd.DataFrame({'x1':x1, 'x2':x2})
y = b1*x1 + b2*x2 + e

# Fit model
lmFit = lm().fit(X, y)
print('intercept = {0:.3f}'.format(lmFit.intercept_))
print(pd.DataFrame(lmFit.coef_, index=X.columns, columns=['coef']))


#%% ### <br>Interaction: numeric variable interact with boolean variable
#' Take note how to interpret the result
from sklearn.preprocessing import PolynomialFeatures as poly

# Simulate data
random.seed(1)
n = 10000
x1 = np.array(['N','Y']*int(n/2))
x1 = np.random.choice(x1, n, replace=False)
x2 = np.random.uniform(0, 1, n)
e = np.random.uniform(-0.1, 0.1, n)

bn = -2
by = 1

# Covert categorical variable
X = pd.DataFrame({'x1':x1, 'x2':x2})
X = pd.get_dummies(X, columns=['x1'], drop_first=True)  # x2, x1_Y

# Interaction
X = pd.DataFrame(poly(interaction_only=True, include_bias=False).fit_transform(X),
                 columns=np.append(X.columns.values, 'x1_Y:x2')) # x2, x1_Y, x2:x1_Y

y = np.where(x1=='N', bn*x2, by*x2) + e

# Fit model
lmFit = lm().fit(X, y)
print('intercept = {0:.3f}'.format(lmFit.intercept_))
print(pd.DataFrame(lmFit.coef_, index=X.columns, columns=['coef']))

#%% ### <br>Polynomial numeric variable
from sklearn.preprocessing import PolynomialFeatures as poly

# Simulate data
random.seed(1)
n = 10000
x1 = np.random.uniform(2, 5, n)
e = np.random.uniform(-0.5, 0.5, n)

b1 = 2
b2 = 3

xPoly = poly(2, include_bias=False).fit_transform(np.array([[x, 1] for x in x1]))
X = pd.DataFrame(np.array([v[[0,2]] for v in xPoly]),
                 columns=['x1', 'x1^2'])

y = b2*X['x1^2'] + b1*X['x1'] + e

# Fit model
lmFit = lm().fit(X, y)
print('intercept = {0:.3f}'.format(lmFit.intercept_))
print(pd.DataFrame(lmFit.coef_, index=X.columns, columns=['coef']))

#%% # <br>Example: Classification
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

dfTrain0 = pd.read_csv("../0. Data/(2016 UCI) Credit Default/data_train.csv")
dfTest0 = pd.read_csv("../0. Data/(2016 UCI) Credit Default/data_test.csv")

def prepTrainTest(df):
    df = pd.get_dummies(df, columns=['Default', 'Sex', 'Marriage'], drop_first=True)
    df = pd.get_dummies(df, columns=['Education', 'SepRepayment']).drop(['Education_high school', 'SepRepayment_paid'], axis=1)
    return df
dfTrain = prepTrainTest(dfTrain0)
XTrain = dfTrain.drop('Default_1',axis=1)
yTrain = dfTrain['Default_1']

# Fit model
## Note: LogisticRegression() forces L1 or L2 regularization. To remove regularization,
## need to set penalty to 'l1' and C to a large value.
lmModel = LogisticRegression(class_weight="balanced", penalty='l1', C=1000)
lmFit = lmModel.fit(XTrain, yTrain)
print('intercept = {0:.3f}'.format(lmFit.intercept_[0]))
print(pd.DataFrame(lmFit.coef_.T, index=XTrain.columns, columns=['coef']))

#%% ### <br>Train performance
probTrain = [x[1] for x in lmFit.predict_proba(XTrain)]
predTrain = lmFit.predict(XTrain)

cmTrain = pd.DataFrame(confusion_matrix(yTrain, predTrain))
cmTrain.columns = pd.Series(cmTrain.columns).apply(lambda s: 'pred'+str(s))
cmTrain.index = pd.Series(cmTrain.index).apply(lambda s: 'actual'+str(s))
cmTrain

perfTrain = pd.DataFrame({'F1':[round(f1_score(yTrain, predTrain), 3)],
                          'AUC':[round(roc_auc_score(yTrain, probTrain),3)]})
print(perfTrain)

#%% ### <br>Cross-Validation performance
scores = cross_val_score(lmModel, XTrain, yTrain, scoring='f1', cv=10)

print('Cross-validation f1 score is {0:.3f}'.format(scores.mean()))

#%% ### <br>Test performance
dfTest = prepTrainTest(dfTest0)
XTest = dfTest.drop('Default_1',axis=1)
yTest = dfTest['Default_1']
    
predTest = lmFit.predict(XTest)

cmTest = pd.DataFrame(confusion_matrix(yTest, predTest))
cmTest.columns = pd.Series(cmTest.columns).apply(lambda s: 'pred'+str(s))
cmTest.index = pd.Series(cmTest.index).apply(lambda s: 'actual'+str(s))
cmTest

f1Test = f1_score(yTest, predTest)
print('Test f1 score = {0:.3f}'.format(f1Test))

#%% # <br>Appendix: Wrap statsmodels in sklearn estimator
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

dfTrain0 = pd.read_csv("../0. Data/(2016 UCI) Credit Default/data_train.csv")
dfTest0 = pd.read_csv("../0. Data/(2016 UCI) Credit Default/data_test.csv")

def prepTrainTest(df):
    df = pd.get_dummies(df, columns=['Default', 'Sex', 'Marriage'], drop_first=True)
    df = pd.get_dummies(df, columns=['Education', 'SepRepayment']).drop(['Education_high school', 'SepRepayment_paid'], axis=1)
    return df
dfTrain = prepTrainTest(dfTrain0)
XTrain = dfTrain.drop('Default_1',axis=1)
yTrain = dfTrain['Default_1']
XTrain = sm.add_constant(XTrain)

w = sum(yTrain==0)/sum(yTrain==1)
yWeights = yTrain.apply(lambda x: w if x==1 else 1)

# Fit model
lmModel = sm.GLM(yTrain, XTrain, family=sm.families.Binomial(), freq_weights=yWeights)
lmFit = lmModel.fit()
lmFit.params

#%% ### <br>Train performance
probTrain = lmFit.predict(XTrain)
predTrain = probTrain.apply(lambda x: 0 if x<0.5 else 1)

cmTrain = pd.DataFrame(confusion_matrix(yTrain, predTrain))
cmTrain.columns = pd.Series(cmTrain.columns).apply(lambda s: 'pred'+str(s))
cmTrain.index = pd.Series(cmTrain.index).apply(lambda s: 'actual'+str(s))
cmTrain

prfsTrain = precision_recall_fscore_support(yTrain, predTrain, average='binary', pos_label=1)
prfsTrain = {"precision": prfsTrain[0],
             "recall": prfsTrain[1],
             "f1-score": prfsTrain[2],
             "support": prfsTrain[3]}
prfsTrain

#%% ### <br>Cross-Validation performance
#' Wrap statsmodels GLM to use cross_val_score from sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticsRegression(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return

    def fit(self, X, y):
        ys = y.unique()
        ys.sort()
        w = sum(y==ys[0])/sum(y==ys[1])
        yWegiths = y.apply(lambda x: w if x==ys[1] else 1)
        self.fitted = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=yWegiths).fit()   
        return self
    
    def predict(self, X):
        prob = self.fitted.predict(X)
        pred = prob.apply(lambda x: 0 if x<0.5 else 1)
        return pred

scores = cross_val_score(LogisticsRegression(), XTrain, yTrain, scoring='f1', cv=10)

print('Cross-validation f1 score is {0:.3f}'.format(scores.mean()))

#%% ### <br>Test performance
dfTest = prepTrainTest(dfTest0)
XTest = dfTest.drop('Default_1',axis=1)
yTest = dfTest['Default_1']
XTest = sm.add_constant(XTest)
    
probTest = lmFit.predict(XTest)
predTest = probTest.apply(lambda x: 0 if x<0.5 else 1)

cmTest = pd.DataFrame(confusion_matrix(yTest, predTest))
cmTest.columns = pd.Series(cmTest.columns).apply(lambda s: 'pred'+str(s))
cmTest.index = pd.Series(cmTest.index).apply(lambda s: 'actual'+str(s))
cmTest

prfsTest = precision_recall_fscore_support(yTest, predTest, average='binary', pos_label=1)
prfsTest = {"precision": prfsTest[0],
             "recall": prfsTest[1],
             "f1-score": prfsTest[2],
             "support": prfsTest[3]}
prfsTest










