#!/usr/bin/env python
# coding: utf-8

# ### Project

# #### 1. Import the datasets and libraries, check datatype, statistical summary, shape, null values or incorrect imputation.

# ##### Import Libraries

# In[169]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import zscore,skew,norm,probplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# ##### Load datasets and review

# In[170]:


loan = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
loan.shape
loan.head()


# In[171]:


loan.describe()


# In[172]:


loan.info()


# In[173]:


loan.isnull().sum().sum()


# In[174]:


loan


# In[175]:


loan[~loan.applymap(np.isreal).all(1)]  #checking for incorrect imputations


# #### 2. EDA: Study the data distribution in each attribute and target variable, share your findings
# #### (20 marks)
#  Number of unique in each column?
#  Number of people with zero mortgage?
#  Number of people with zero credit card spending per month?
#  Value counts of all categorical columns
#  Univariate and Bivariate
#  Get data model ready

# In[176]:


loan.nunique(axis = 0, dropna = True)         # Number of unique in each column


# In[177]:


loan[loan['Mortgage']==0].Mortgage.count()          # Number of people with zero mortgage


# In[178]:


loan[loan['CCAvg']==0].CCAvg.count()              # Number of people with zero credit card spending per month


# In[179]:


loan['Education'] = loan['Education'].replace({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'})
loan.head()
loan['Education'].value_counts()  # Value counts of all categorical columns. Here Education is a categorical column


# In[180]:


loan['Experience'].value_counts()


# In[181]:


loan['Family'].value_counts()


# In[222]:


loan['Age'].value_counts()


# In[182]:


loan['Personal Loan'].value_counts()


# In[183]:


loan['Securities Account'].value_counts()


# In[184]:


loan['CD Account'].value_counts()


# In[185]:


loan['Online'].value_counts()


# In[186]:


loan['CreditCard'].value_counts()


# In[187]:


# Create dummy variables for education to work on it

loan = pd.get_dummies(loan, columns=['Education'])
loan.head()


# In[188]:


loan[loan['Experience']<0]['Experience'].value_counts()  #checking for negative values of experience and finding the unique count of each


# In[189]:


#these negative values are incorrect and need to be replaced
#lets check the corr of various attributes 
loan.corr()


# In[190]:


plt.figure(figsize=(10,7))      # using heatmap to get a better idea about the correation
sns.heatmap(loan.corr(), annot=True)   


# In[191]:


## We can see high correlation between Age and Experience. Thus we can replace negative values of experience with mean/median
## of the positive experience values having that age corresponding to negative exp.
loan[loan['Experience']<0]['Age'].value_counts()


# In[192]:


# For all -1 Experience

age=loan[loan['Experience']==-1]['Age'].unique().tolist()
exp_index=loan[loan['Experience']==-1]['Experience'].index.tolist()
cond = loan[(loan['Age'].isin(age)) & (loan['Experience'] > 0)] #rows and columns that has age equal to that with experience = minus one, AND, where experience is > 0   
#if the above condition is true we need to find mean of experience in that case
for i in exp_index:
   loan.loc[i,'Experience'] = cond['Experience'].mean()
loan['Experience']=loan.Experience.round()


# In[193]:


# For all -2 Experience

age2=loan[loan['Experience']==-2]['Age'].unique().tolist()
exp_index2=loan[loan['Experience']==-2]['Experience'].index.tolist()
cond2 = loan[(loan['Age'].isin(age2)) & (loan['Experience'] > 0)] #rows and columns that has age equal to that with experience = minus one, AND, where experience is > 0   
#if the above condition is true we need to find mean of experience in that case
for i in exp_index2:
   loan.loc[i,'Experience'] = cond2['Experience'].mean()
loan['Experience']=loan.Experience.round()


# In[194]:


# For all -3 Experience

age3=loan[loan['Experience']==-3]['Age'].unique().tolist()
exp_index3=loan[loan['Experience']==-3]['Experience'].index.tolist()
cond3 = loan[(loan['Age'].isin(age3)) & (loan['Experience'] > 0)] #rows and columns that has age equal to that with experience = minus one, AND, where experience is > 0   
#if the above condition is true we need to find mean of experience in that case
for i in exp_index3:
   loan.loc[i,'Experience'] = cond3['Experience'].mean()
loan['Experience']=loan.Experience.round()


# In[195]:


loan[loan['Experience']<0].sum().sum()  # checking for any left out negative values in Experience colum


# ###### Univariate and Bivariate

# In[196]:


columns = list(loan)[:] # Excluding Outcome column which has only 
loan[columns].hist(stacked=False, bins=100, figsize=(20,30), layout=(14,2)); 
# Histogram 


# In[197]:


# We will plot the univariate plots of the highly correlated attributes as that is where we might be able to get more valuable insights.
# age,experience,income,ccavg,personal loan
plt.figure(figsize=(10,8))
sns.distplot(loan['Age']);


# In[198]:


plt.figure(figsize=(10,8))
sns.distplot(loan['Experience']);


# In[199]:


plt.figure(figsize=(10,8))
sns.distplot(loan['Income']);


# In[200]:


plt.figure(figsize=(10,8))
sns.distplot(loan['CCAvg']);


# In[201]:


plt.figure(figsize=(10,8))
sns.countplot(loan['Personal Loan']);


# ###### From the above plot we can see that less than 500 people have only taken personal loans.

# In[202]:


## Bivariate relationship plotted using pairplot
plt.figure(figsize=(15,10))
sns.pairplot(loan)
plt.show()

# It can be seen most of them have a distribution which is best suited for fitting a sigmoid curve. 
# Age and Experience displays strong linear relationship.


# #### 3. Split the data into training and test set in the ratio of 70:30 respectively (5 marks)

# In[203]:


x = loan.drop(['Education_Advanced/Professional','Personal Loan'],axis=1)   #set of independent variables
y = loan[['Personal Loan']]                                                 #dependent variable
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=1)   #training and test set for x and y variables


# #### 4.Use Logistic Regression model to predict whether the customer will take personal loan or not. Print all the metrics related for evaluating the model performance (15 marks)

# In[204]:


from sklearn import metrics
import os,sys
from scipy import stats
from sklearn.linear_model import LogisticRegression
# Fit the model on train
modell = LogisticRegression(random_state=42)
modell.fit(x_train, y_train)
#predict on test
y_pred = modell.predict(x_test)


# In[205]:


prop_Y = loan['Personal Loan'].value_counts(normalize=True)
print(prop_Y)


# ### Get your model ready ( part of question.2 ) in the next line

# In[206]:


import statsmodels.api as sm

logit = sm.Logit(y_train, sm.add_constant(x_train))
lg = logit.fit()


# In[207]:


#Summary of logistic regression
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
print(lg.summary())


# #### Pseudo R-square value is 0.6356 which indicates that this is a good model. A pseudo R^2 of 63.56% indicates that 63.56% of the uncertainty of the  model is explained by the full model

# In[208]:


model_score = modell.score(x_test, y_test)
print("Training accuracy",modell.score(x_train,y_train)*100,"%")  
print()
print("Testing accuracy",modell.score(x_test, y_test)*100,"%")


# In[209]:


# function to draw confusion matrix
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()


# ##### Final Model !!!

# In[210]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score,accuracy_score,classification_report
# get accuracy of model
acc_score = accuracy_score(y_test,y_pred)
# get F1-score of model
f1_score = f1_score(y_test,y_pred) 
# get recall-score of model
recll_score = recall_score(y_test,y_pred)
# get precision-score of model
prec_score = precision_score(y_test,y_pred)
# get roc-auc-score of model
rocauc_score = roc_auc_score(y_test,y_pred)
# get the confusion matrix
confmat = draw_cm(y_test,y_pred)
# get the classification report
classrep = classification_report(y_test,y_pred)

print("The accuracy of the model is {} %".format(acc_score*100))
print("The f1-score of the model is {} %".format(f1_score*100))
print("The recall score of the model is: \n",recll_score*100,"%")
print("The precision-score of the model is: \n",prec_score*100,"%")
print("The roc-auc_score is: \n",rocauc_score*100,"%")
print("The confusion matrix for logistic regression is: \n",confmat)
print("Detailed classification report for logistic regression is: \n",classrep)


# In[211]:


#AUC ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, modell.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, modell.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ###### As seen from the above data of the final model, we have accuracy of 90.6% which is a good value. But, it is seen that the recall values and the precision values are not that great, i.e they have low values. Thus the model needs to be improved. 

# #### 5. Give your reasoning on how can the model perform better? (10 marks)
# #### Hint: Check parameter

# In[212]:


# Checking Parameters of logistic regression

model.get_params()


# In[213]:


# Running a loop to check different values of 'solver'
# all solver can be used with l2, only 'liblinear' and 'saga' works with both 'l1' and 'l2'
#Checking for solver values that have best scores (for l2)

train_score=[]
test_score=[]
solver = ['newton-cg','lbfgs','liblinear','sag','saga']
for i in solver:
    model = LogisticRegression(random_state=42,penalty='l2', C = 0.75,solver=i)  # changing values of solver
    model.fit(x_train, y_train) 
    y_predict = model.predict(x_test)     
    train_score.append(round(model.score(x_train, y_train),3))
    test_score.append(round(model.score(x_test, y_test),3))
    
print(solver)
print()
print(train_score)
print()
print(test_score)


# In[214]:


#Checking for solver values that have best scores (for l1)

train_score=[]
test_score=[]
solver = ['liblinear','saga']   # changing values of solver which works with 'l1'
for i in solver:
    model = LogisticRegression(random_state=42,penalty='l1', C = 0.75,solver=i)  #changed penalty to 'l1'
    model.fit(x_train, y_train) 
    y_predict = model.predict(x_test)     
    train_score.append(round(model.score(x_train, y_train),3))
    test_score.append(round(model.score(x_test, y_test),3))
    
print(solver)
print()
print(train_score)
print()
print(test_score)


# #### Highest accuracy is same 'l1' with 'liblinear' and 'l2' with 'newton-cg'
# #### choose any one

# In[223]:


model = LogisticRegression(random_state=42,penalty='l2',solver='newton-cg',class_weight='balanced') # changing class weight to balanced

model.fit(x_train, y_train) 

y_predict = model.predict(x_test)     

print("Training accuracy",model.score(x_train,y_train)*100,"%")  
print()
print("Testing accuracy",model.score(x_test, y_test)*100,"%")


# In[216]:


# Running a loop to check different values of 'C'
# Checking for values of C with best scores

train_score=[]                                 
test_score=[]
C = [0.01,0.1,0.25,0.5,0.75,1]
for i in C:
    model = LogisticRegression(random_state=42,penalty='l1', solver='liblinear',class_weight='balanced', C=i)  # changing values of C
    model.fit(x_train, y_train) 
    y_predict = model.predict(x_test)     
    train_score.append(round(model.score(x_train,y_train),3)) # appending training accuracy in a blank list for every run of the loop
    test_score.append(round(model.score(x_test, y_test),3))   # appending testing accuracy in a blank list for every run of the loop
    
print(C)
print()
print(train_score)
print()
print(test_score)


# #### Best testing accuracy is obtained for C=0.1
# 
# #### Therefore final model is

# In[217]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score,accuracy_score
model = LogisticRegression(random_state=42,penalty='l1', solver='liblinear', class_weight='balanced',C=0.1) 
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("Training accuracy : ",model.score(x_train,y_train)*100,"%")
print()
print("Testing accuracy : ",model.score(x_test, y_test)*100,"%")
print()
print('Confusion Matrix')
print(draw_cm(y_test,y_predict))
print()
print("Recall :", recall_score(y_test,y_predict)*100,"%")
print()
print("Precision :", precision_score(y_test,y_predict)*100,"%")
print()
f1_score = f1_score(y_test,y_predict)
print("F1 Score :{} %".format(f1_score*100))
print()
print("Roc Auc Score :{} %".format(roc_auc_score(y_test,y_predict)*100))


# In[218]:


#AUC ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ##### By making the above changes in the parameters we can see significant increase in accuracy and the scores as compared to the previous
# ##### model. Also it is clear from the AUC ROC Curve that this model is highly preferred over the previous as seen by the threshold and the AUC. 
# ##### AUC is 0.89 (approx. equal to 0.9) in this model which is a very good value as compared to 0.63 from previous model.

# #### 6. Give Business understanding of your model? (5 marks)

# In[219]:


print(draw_cm(y_test,y_predict))
print()


# 
# #### Confusion matrix means
# 
# ##### True Positive (observed=1,predicted=1):
# 
# Predicted that its liability customer buys personal loans and the customer did buy loan
# 
# ##### False Positive (observed=0,predicted=1):
# 
# Predicted that its liability customer buys personal loans and the customer did not buy loan
# 
# ##### True Negative (observed=0,predicted=0):
# 
# Predicted that its liability customer does not buy personal loans and the customer did not buy loan
# 
# ##### False Negative (observed=1,predicted=0):
# 
# Predicted that its liability customer does not buy personal loans and the customer did buy loan
# 
# Here the bank wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors) 
# ###### i.e. more number of True Positive, if TP is high bank would make more profit. 
# ###### Hence Recall is the important metric.
# 
# In case of True negative bank will lose few customers but that okay because the bank would want to retain money more than customers who are not actually taking up the loan.
# 
# Also we look onto minimising false positive since we dont want our prediction to go wrong.
# In case of False negative, its not a problem as the customers end up taking loan from the bank.
# 
# After achieving the desired accuracy we can deploy the model for practical use. As in the bank now can predict its liability customers converting to personal loan customers. They can use the model for upcoming customers.
