
# coding: utf-8

# In[237]:


get_ipython().system('python --version')


# In[238]:


pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)


# In[239]:


import pandas as pd
import numpy as np


# In[240]:


HDLo = pd.read_csv('HDLData1.csv')
HDLo


# In[241]:


#EDA
#column names
HDLo.columns

'''
Areaname – city / town  name
County – zipcode
State 
Lcount – count of Lowes stores in the town/city
Hdcount – count of Home Depot in the town/city
Pop_2000 – population in 2000
Pop_2010 – population in 2010
Income_2000 – avg income in 2000
Income_2010 – avg income in 2010
pct_U18_2000 – percent under 18 in 2000 
pct_U18_2010 – percent under 18 in 2010
Pct_college_2000 – percent in college per town in 2000
Pct_college_2010 – percent in college per town in 2010
Ownhome_2000 – percent owned home in 2000
Ownhome_2010 – percent owned home in 2010
Density_2000 – percent density per town in 2000
Density_2010 – percent density per town in 2010
Pct_white_2000 – percent of Caucasian in town in 2000
Pct_white_2010 – percent of Caucasian in town in 2010
Pct_black_2000 – percent African American in 2000
Pct_black_2010 – percent African American in 2010
'''



# In[242]:


#stats on the columns
HDLo.describe()


# In[243]:


#dimension on dataframe
HDLo.shape


# In[244]:


#find missing values
# reference https://medium.com/@numanyilmaz61/handling-missing-data-93d3ce5d0161
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(HDLo)



# In[245]:



#remove NAs
HDLo = HDLo.dropna()


# In[246]:


#Question 1 

#1.	Create dummy variables to identify if HomeDepot or Lowes is present in that county
## reference https://towardsdatascience.com/the-dummys-guide-to-creating-dummy-variables-f21faddb1d40

HDLo['HD_present'] = np.where(HDLo['HDcount'] > 0, 1, 0)
HDLo['Lo_present'] = np.where(HDLo['Lcount'] > 0, 1,0)





# In[261]:





# In[247]:


#pandas profile 
import pandas_profiling
pandas_profiling.ProfileReport(HDLo)


# In[260]:


#using seaborn for correlation
import seaborn as sns
corr = HDLo.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[249]:


HDLo.columns


# In[250]:


# reset the index
HDLo = HDLo.reset_index() 


# In[251]:


#prediction for HD stores 

import pandas as pd
import statsmodels.api as sm

X = HDLo.filter(['Lo_present','pop_2000' ,     
           'pop_2010' , 'income_2000' , 'income_2010' , 'pct_U18_2000' ,   
           'pct_U18_2010' , 'pctcollege_2000' , 'pctcollege_2010' , 'ownhome_2000' ,   
           'ownhome_2010' , 'density_2000' , 'density_2010' , 'pctwhite_2000' ,
           'pctwhite_2010' , 'pctblack_2000' , 'pctblack_2010'])

y = HDLo.filter(['HD_present'])

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.externals import joblib
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

#scoring the model
print('Accuracy score: ')
print(accuracy)
print(classification_report(y_test, predictions))

#Get ALL results from the model
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# accuracy of coeficants
logit = sm.Logit(y, X)
logit.fit().params

#cross validation matrix of accuracy
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
# fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix)
# plt.show()

#save model to use on HD's without store
filename = 'HD_predic.sav'
joblib.dump(logreg, filename)


# In[252]:


loaded_model = joblib.load(filename)

#filter and then predict on stores where HD is not present 
HD_no = HDLo[(HDLo.HDcount == 0)]

HD_X = HD_no.filter(['Lo_present','pop_2000' ,     
           'pop_2010' , 'income_2000' , 'income_2010' , 'pct_U18_2000' ,   
           'pct_U18_2010' , 'pctcollege_2000' , 'pctcollege_2010' , 'ownhome_2000' ,   
           'ownhome_2010' , 'density_2000' , 'density_2010' , 'pctwhite_2000' ,
           'pctwhite_2010' , 'pctblack_2000' , 'pctblack_2010'])
HD_predict = loaded_model.predict_proba(HD_X)
HD_predict = pd.DataFrame(HD_predict)
HD_predict = HD_predict.round(4)
New_HD_stores = pd.concat([HD_predict.reset_index(drop=True), HD_no.reset_index(drop=True)], axis=1)
New_HD_stores = New_HD_stores.sort_values(by=[1], ascending=False)


# In[253]:


New_HD_stores


# In[254]:


#prediction for Lo stores 

import pandas as pd
import statsmodels.api as sm

X = HDLo.filter(['HD_present','pop_2000' ,     
           'pop_2010' , 'income_2000' , 'income_2010' , 'pct_U18_2000' ,   
           'pct_U18_2010' , 'pctcollege_2000' , 'pctcollege_2010' , 'ownhome_2000' ,   
           'ownhome_2010' , 'density_2000' , 'density_2010' , 'pctwhite_2000' ,
           'pctwhite_2010' , 'pctblack_2000' , 'pctblack_2010'])

y = HDLo.filter(['Lo_present'])

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.externals import joblib
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg_l = LogisticRegression()
logreg_l.fit(X_train, y_train)
predictions = logreg_l.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

#scoring the model
print('Accuracy score: ')
print(accuracy)
print(classification_report(y_test, predictions))

#Get ALL results from the model
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# accuracy of coeficants
logit = sm.Logit(y, X)
logit.fit().params

#cross validation matrix of accuracy
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
# fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix)
# plt.show()

#save model to use on HD's without store
filename2 = 'Lo_predic.sav'
joblib.dump(logreg_l, filename2)


# In[259]:


loaded_model = joblib.load(filename2)

#filter and then predict on stores where LO is not present 
Lo_no = HDLo[(HDLo.Lo_present == 0)]

Lo_X = HD_no.filter(['Lo_present','pop_2000' ,     
           'pop_2010' , 'income_2000' , 'income_2010' , 'pct_U18_2000' ,   
           'pct_U18_2010' , 'pctcollege_2000' , 'pctcollege_2010' , 'ownhome_2000' ,   
           'ownhome_2010' , 'density_2000' , 'density_2010' , 'pctwhite_2000' ,
           'pctwhite_2010' , 'pctblack_2000' , 'pctblack_2010'])
Lo_predict = loaded_model.predict_proba(Lo_X)
Lo_predict = pd.DataFrame(Lo_predict)
Lo_predict = Lo_predict.round(4)
New_Lo_stores = pd.concat([Lo_predict.reset_index(drop=True), Lo_no.reset_index(drop=True)], axis=1)
New_Lo_stores = New_Lo_stores.sort_values(by=[1], ascending=False)
New_Lo_stores


# In[256]:


Lo_no

