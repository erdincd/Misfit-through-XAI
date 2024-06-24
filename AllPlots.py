#Installing and importing the required libraries and modules
!pip install numpy pandas scikit-learn shap dtreeviz matplotlib pyomo opticl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
from sklearn.metrics import f1_score
import dtreeviz
import matplotlib.pyplot as plt
from sklearn import tree
import random
from sklearn.utils import check_random_state
import embed_mip as em
import ce_helpers as ch

#Please set the number of additional variables in your data below
t=1 

#Please set the class names
class_names=['FIT','MISFIT']

#Setting random seeds
rn=42
random.seed(rn)
np.random.seed(rn)
random_state=check_random_state(rn)

#Importing the data
data= pd.read_csv("Data.csv")
out=data.columns[-1]
n=len(data.columns)-1
label1=data.columns[0][0]
label2=data.columns[n-t-1][0]
X=data[data.columns[:-1-t]]
y=data[out]

#Defining the models to be built
MODELS=[]
clfPJ2=DecisionTreeClassifier(max_depth=2, random_state=rn)
clfPJ3=DecisionTreeClassifier(max_depth=3, random_state=rn)
clfPJ4=DecisionTreeClassifier(max_depth=4, random_state=rn)
clfPJ5=DecisionTreeClassifier(max_depth=5, random_state=rn)
clfPJ6=DecisionTreeClassifier(max_depth=6, random_state=rn)
clfPJ7=DecisionTreeClassifier(max_depth=7, random_state=rn)
clfPJ8=DecisionTreeClassifier(max_depth=8, random_state=rn)
clfPJ9=DecisionTreeClassifier(max_depth=9, random_state=rn)
clfPJ10=DecisionTreeClassifier(max_depth=10, random_state=rn)
clfPJ= DecisionTreeClassifier(random_state=rn)
clfPJ2_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=2, random_state=rn)
clfPJ3_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=3, random_state=rn)
clfPJ4_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=4, random_state=rn)
clfPJ5_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=5, random_state=rn)
clfPJ6_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=6, random_state=rn)
clfPJ7_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=7, random_state=rn)
clfPJ8_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=8, random_state=rn)
clfPJ9_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=9, random_state=rn)
clfPJ10_min5=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, random_state=rn)
clfPJ_min5= DecisionTreeClassifier(min_samples_leaf=5, random_state=rn)
gbr= GradientBoostingClassifier(n_estimators=10000, max_depth=5, random_state=rn)
linr=LogisticRegression()
MODELS.extend([clfPJ2, clfPJ3, clfPJ4, clfPJ5, clfPJ6, clfPJ7, clfPJ8, clfPJ9, clfPJ10, clfPJ, clfPJ2_min5,
               clfPJ3_min5, clfPJ4_min5, clfPJ5_min5, clfPJ6_min5, clfPJ7_min5, clfPJ8_min5, clfPJ9_min5,
               clfPJ10_min5, clfPJ_min5, gbr, linr])

#Fitting the models to the data
R2=[]
PRED=[]
for i in range(len(MODELS)):
  MODELS[i].fit(X, y)
  PRED.append(MODELS[i].predict(X))
  R2.append(f1_score(y,PRED[i],average='macro'))
R2

#Please set the colors of Logistic Regression Model below
color_pos='dimgrey'
color_neg='lightgray'

#Plotting Logistic Regression Model results
def tornado_plot(attributes, values1, values2, title):
  scores=[abs(v1+v2) for v1, v2 in zip(values1, values2)]
  sorted_indices=np.argsort(scores)[::-1]
  attributes=[attributes[i] for i in sorted_indices]
  values1=[values1[i] for i in sorted_indices]
  values2=[values2[i] for i in sorted_indices]
  fig, ax=plt.subplots()
  attribute_positions=np.arange(len(attributes))
  ax.barh(attribute_positions-0.2, values1, height=0.4, color=color_pos, label=label1)
  ax.barh(attribute_positions+0.2, values2, height=0.4, color=color_neg, label=label2)
  ax.set_yticks(attribute_positions)
  ax.set_yticklabels(attributes)
  ax.invert_yaxis()  
  for i in range(len(attributes)-1):
    ax.axhline(attribute_positions[i]+0.5, color='gray', linestyle='dashed', linewidth=1)
  ax.set_xlabel('Coefficients')
  ax.set_title(title)
  ax.legend()
  plt.show()
labels=[col.split('-')[1] for col in data.columns if col.startswith(label1+'-')]
all_coefs=linr.coef_[0]
pos=all_coefs[:len(labels)]
neg=all_coefs[len(labels):2*len(labels)]
tornado_plot(labels, pos, neg, title="Logistic Regression Model")

#Please set the depth level of decision tree you want to plot (set -1 for unlimited depth),
#and with or without minimum number of samples per node (1: with, 0:without)
depth=4
with_min5=1

#Plotting Decision Tree Model results
if depth==-1:
  depth==11
DTmodel=MODELS[depth-2+10*with_min5]
viz_model=dtreeviz.model(DTmodel, X_train=X, y_train=y,
                           feature_names=[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')], 
                           class_names=class_names)
v=viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15) 
v.show()

#Please set the index of the data point you want to see its SHAP results
ind=21

#Plotting SHAP results
explainer=shap.Explainer(gbr.predict, X, seed=rn)
shap_values=explainer(X)
shap.plots.waterfall(shap_values[ind])
shap.summary_plot(shap_values, X,
                  feature_names=[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')])

#Please set the index of the data point you want to see its CE results and the number of CEs
ind_2=21
num_counterfactuals=5

#Computing Counterfactual Explanations (you can see the alternative solutions in df_1 variable)
y_pred=np.ones(len(y))
y_temp=1-y
d={'target': out, 'numerical':[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')]}
version='Data_v1'
alg_list_cf=['gbm']
alg_list_dr=['mlp']
alg='gbm'
algorithms={'counterfactual_german':alg}
while y_pred[ind_2]==1:
  y_temp=1-y_temp
  outcome_dict={'counterfactual_german':{'task': 'binary', 'X features': X.columns, 'class': d['target'], 'alg_list': alg_list_cf,
                                           'X_train':X, 'X_test':X, 'y_train':y_temp, 'y_test':y_temp}}
  ch.train_models(outcome_dict, version)
  performance=ch.perf_trained_models(version, outcome_dict)
  y_pred, y_pred_0, X_test_0, models=ch.load_model(algorithms, outcome_dict, 'counterfactual_german')
clf=models['counterfactual_german']
algorithm=algorithms['counterfactual_german']
constraints_embed=['counterfactual_german']
objectives_embed={}
model_master=em.model_selection(performance[performance['alg']==algorithm], constraints_embed, objectives_embed)
model_master['lb']=0.5
model_master['ub']=None
model_master['SCM_counterfactuals']=None
model_master['features']=[[col for col in X.columns]]
y_ix_1=np.where(y==1)
X1=X.iloc[y_ix_1[0],:].copy().reset_index(drop=True, inplace=False)
CEs, CEs_, final_model=ch.opt(X, X1, X.loc[ind_2], d['numerical'], [], [], {}, [], [], [], X.columns, 
                                        True, 0, False, False, num_counterfactuals, model_master, None)
df_1=ch.visualise_changes(clf, d, F_coh={}, method='CE-OCL', CEs=CEs, CEs_=CEs_, only_changes=True)

#Please set the depth level of decision tree you want to plot (set -1 for unlimited),
#and the minimum number of samples per node (set -1 for unlimited)
depth=4
minspn=5

#Fitting the Decision Tree model to the data with additional variables
if depth>0 and minspn>0:
  clfAV=DecisionTreeClassifier(min_samples_leaf=minspn, max_depth=depth, random_state=42)
elif depth>0:
  clfAV=DecisionTreeClassifier(max_depth=depth, random_state=rn)
elif minspn>0:
  clfAV=DecisionTreeClassifier(min_samples_leaf=minspn, random_state=rn)
else:
  clfAV=DecisionTreeClassifier(random_state=rn)
X= data[data.columns[:-1]]
y= data[out]
clfAV.fit(X, y)
clfAV.predict(X.values)
R2.append(f1_score(y,PRED[len(PRED)-1],average='macro'))

#Plotting Decision Tree model (with additional variables) results
viz_model=dtreeviz.model(clfAV, X_train=X, y_train=y, class_names=class_names,
                           feature_names=[col for col in data.columns if col!=out])
v=viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15) 
v.show()
