import numpy as np
import pandas as pd
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

#class1 data
mean1=np.random.randint(5,15,       #range from 5 to 15
                       (300,8))     #data size
variance1=np.random.normal(loc=1,scale=1,size=2400).reshape(300,8)    #return value in normal distribution
#specify this have 8 features,300 samples
#np.random.rand（） just return a value from 0 to 1
features1=mean1+variance1
label1=[]
for i in range(300):
    label1.append('1')
lab1=np.array(label1).T
class1=np.insert(features1,8, values=lab1, axis=1)  #add the label as the last column of the data
#insert lab1 into features in 8th columns,in the axis=1 direction(column direction)

#class2 data
mean2=np.random.randint(15,25,       #range from 15 to 25
                    (300,8))         #data size
variance2=np.random.normal(loc=1,scale=1,size=2400).reshape(300,8)      #return value in normal distribution
features2=mean2+variance2
label2=[]
for i in range(300):
    label2.append('2')
lab2=np.array(label2).T
class2=np.insert(features2,8, values=lab2, axis=1)

#class3 data
mean3=np.random.randint(25,35,       #range from 25 to 35
                    (300,8))         #data size
variance3=np.random.normal(loc=1,scale=1,size=2400).reshape(300,8)      #return value in normal distribution
features3=mean3+variance3
label3=[]
for i in range(300):
    label3.append('3')
lab3=np.array(label3).T
class3=np.insert(features3,8, values=lab3, axis=1)
#print(np.min(features1))       #check if there is any negative value,or we can't use selectKbest function

#merge all the data
all_features=np.concatenate((features1,features2,features3),axis=0)
all_label=np.concatenate((lab1,lab2,lab3),axis=0)
all_class=np.concatenate((class1,class2,class3),axis=0)

#set some columns to zeros
clear_column=[]    #the columns we need to clean
zero=np.zeros([900,8])
#clean some class data column
for i in clear_column:
    zero[:,i]+=all_features[:,i]
final_features=all_features-zero

#clean those unimportant columns
sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
sel.fit_transform(final_features)
print(final_features.shape)
#remove all features that are either one or zero (on or off)
# in more than 80% of the samples

#select the best k functions,k varies from 1 to 8
one_feature = SelectKBest(chi2, k=8).fit_transform(final_features,all_label)
#Univariate feature selection works by selecting the best K features
# based on univariate statistical tests.
# It can be seen as a preprocessing step to an estimator.

###consider their coefficient of variation###
corrcoef=np.corrcoef(all_class)
#Return Pearson product-moment correlation coefficients
#print(corrcoef)
np.savetxt("corrcoef.csv", corrcoef, delimiter=",")

cov=np.cov(all_class)
#Covariance indicates the level to which two variables vary together
#examine N-dimensional samples, X = [x_1, x_2, ... x_N]^T,
#the covariance matrix element C_{ij} is the covariance of x_i and x_j
#print(cov)
np.savetxt("cov.csv", cov, delimiter=",")

### establish the model###
parameters = {'n_estimators': [50, 60, 90, 120, 150]}

results_df = pd.DataFrame(columns=['Iteration Times','Accuracy (%)', 'Time (s)'])

for i in list(parameters.values())[0]:
    clf=AdaBoostClassifier(DecisionTreeClassifier(),
                            n_estimators=i,
                            algorithm='SAMME',
                            learning_rate=1)
    start=time.time()
    clf.fit(one_feature,all_label)
    end=time.time()
    duration=end-start
    results_df.loc[i, 'Iteration Times'] = i
    results_df.loc[i, 'Accuracy (%)'] = clf.score(one_feature,all_label)*100
    results_df.loc[i, 'Time (s)'] = duration
    print('time consumption:{:.3f}s'.format(duration))
    print('model accuracy:{:3f}%'.format(clf.score(one_feature,all_label)*100))
    joblib.dump(clf, "train_model_par{:1d}.m".format(i))
    # save the model,with different name
results_df.to_csv('estimators_outcome_parameters.csv',index=0)  #save the outcome,no index

