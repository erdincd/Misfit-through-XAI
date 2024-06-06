#Importing the required libraries and modules
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

#Please set the number of input variables in your data below 
n=37

#Please set the number of additional variables in your data below
t=1 

#Please set the output variable name below as it is written in data
out="PO"

#Please set the person and environment names below
label1='P'
label2='O'

#Please set the colors of Logistic Regression Model below
color_pos='dimgrey'
color_neg='lightgray'

#Please set the class names
class_names=['FIT','MISFIT']

#Setting random seeds
random.seed(42)
np.random.seed(42)
random_state=check_random_state(42)

#Importing the data
raw_data= pd.read_csv("Data.csv")
data_inp=raw_data[raw_data.columns[0:n]]
data=data_inp.copy()
data.insert(n, out, raw_data[out])
data=data.dropna(axis=0, how='any', subset=None, inplace=False)

#Please set the indices of the data points to be in the test data, or set "autosplit" 1 to make code set for you
#Please set test_size if you select auto split
autosplit=0
test_size=0.2
test_indices=[5, 9, 16, 20, 23, 24, 35, 42, 46, 47, 56, 59, 61, 68, 70, 72, 73, 78, 82, 85, 91, 98,
              110, 115, 122, 126, 130, 132, 134, 141, 143, 149, 158, 170, 171, 177, 182, 185, 190, 192,
              198, 202, 208, 213, 219, 220, 223, 230, 239, 240, 247]

#Splitting the data into test and train data (without additional variables)
if autosplit == 1:
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
else:
    test_data=data.iloc[test_indices]
    train_data=data.drop(test_indices)
X_train = train_data[train_data.columns[:-1-t]]
y_train = train_data[out]
X_test = test_data[test_data.columns[:-1-t]]
y_test = test_data[out]

#Defining the models to be built
MODELS=[]
clfPJ2 = DecisionTreeClassifier(max_depth=2, random_state=42)
clfPJ3 = DecisionTreeClassifier(max_depth=3, random_state=42)
clfPJ4 = DecisionTreeClassifier(max_depth=4, random_state=42)
clfPJ5 = DecisionTreeClassifier(max_depth=5, random_state=42)
clfPJ6 = DecisionTreeClassifier(max_depth=6, random_state=42)
clfPJ7 = DecisionTreeClassifier(max_depth=7, random_state=42)
clfPJ8 = DecisionTreeClassifier(max_depth=8, random_state=42)
clfPJ9 = DecisionTreeClassifier(max_depth=9, random_state=42)
clfPJ10 = DecisionTreeClassifier(max_depth=10, random_state=42)
clfPJ= DecisionTreeClassifier(random_state=42)
clfPJ2_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=2, random_state=42)
clfPJ3_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=3, random_state=42)
clfPJ4_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=4, random_state=42)
clfPJ5_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=5, random_state=42)
clfPJ6_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=6, random_state=42)
clfPJ7_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=7, random_state=42)
clfPJ8_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=8, random_state=42)
clfPJ9_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=9, random_state=42)
clfPJ10_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=10, random_state=42)
clfPJ_min5= DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
gbr= GradientBoostingClassifier(n_estimators=10000, max_depth=5, random_state=42)
linr = LogisticRegression()
MODELS.extend([clfPJ2, clfPJ3, clfPJ4, clfPJ5, clfPJ6, clfPJ7, clfPJ8, clfPJ9, clfPJ10, clfPJ, clfPJ2_min5, 
               clfPJ3_min5, clfPJ4_min5, clfPJ5_min5, clfPJ6_min5, clfPJ7_min5, clfPJ8_min5, clfPJ9_min5, 
               clfPJ10_min5, clfPJ_min5, gbr, linr])

#Fitting the models to the data
R2=[]
PRED=[]
for i in range(len(MODELS)):
  MODELS[i].fit(X_train, y_train)
  PRED.append(MODELS[i].predict(X_test))
  R2.append(f1_score(y_test,PRED[i],average='macro'))

#Plotting Logistic Regression Model results
def tornado_plot(attributes, values1, values2, title):
  scores = [abs(v1 + v2) for v1, v2 in zip(values1, values2)]
  sorted_indices = np.argsort(scores)[::-1]
  attributes = [attributes[i] for i in sorted_indices]
  values1 = [values1[i] for i in sorted_indices]
  values2 = [values2[i] for i in sorted_indices]
  fig, ax = plt.subplots()
  attribute_positions = np.arange(len(attributes))
  ax.barh(attribute_positions - 0.2, values1, height=0.4, color=color_pos, label=label1)
  ax.barh(attribute_positions + 0.2, values2, height=0.4, color=color_neg, label=label2)
  ax.set_yticks(attribute_positions)
  ax.set_yticklabels(attributes)
  ax.invert_yaxis()  
  for i in range(len(attributes) - 1):
    ax.axhline(attribute_positions[i] + 0.5, color='gray', linestyle='dashed', linewidth=1)
  ax.set_xlabel('Coefficients')
  ax.set_title(title)
  ax.legend()
  plt.show()
labels = [col.split('-')[1] for col in data.columns if col.startswith(label1+'-')]
all_coefs = linr.coef_[0]
pos = all_coefs[:len(labels)]
neg = all_coefs[len(labels):2*len(labels)]
tornado_plot(labels, pos, neg, title="Logistic Regression Model")

#Please set the depth level of decision tree you want to plot (set -1 for unlimited depth),
#and with or without minimum number of samples per node (1: with, 0:without)
depth=4
with_min5=1
if depth==-1:
  depth==11

#Plotting Decision Tree Model results
DTmodel=MODELS[depth-2+10*with_min5]
viz_model = dtreeviz.model(DTmodel, X_train=X_train, y_train=y_train,
                           feature_names=[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')], 
                           class_names=class_names)
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15) 
v.show()

#Please set the index of the data point (in test data) you want to see its SHAP results
ind=202

#Plotting SHAP results
explainer = shap.Explainer(gbr.predict, X_test, seed=123456789)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[test_indices.index(ind)])
shap.summary_plot(shap_values, X_test,
                  feature_names=[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')])

#Computing Counterfactual Explanations
y_train=1-y_train
y_test=1-y_test
d = {
    'target': out,
    'numerical':[col for col in data.columns if col.startswith(label1+'-') or col.startswith(label2+'-')]
}
version = 'Data_v1'
alg_list_cf = ['gbm']
alg_list_dr = ['mlp']
outcome_dict = {'counterfactual_german':{'task': 'binary', 'X features': X_train.columns, 
                                        'class': d['target'], 'alg_list': alg_list_cf,
                                        'X_train':X_train, 'X_test':X_test,
                                        'y_train':y_train, 'y_test':y_test}}
import embed_mip as em
import ce_helpers
ce_helpers.train_models(outcome_dict, version)
performance = ce_helpers.perf_trained_models(version, outcome_dict)
performance
alg='gbm'
algorithms = {'counterfactual_german':alg}
y_pred, y_pred_0, X_test_0, models = ce_helpers.load_model(algorithms, outcome_dict, 'counterfactual_german')
X_test_0.head()
clf = models['counterfactual_german']
F_r = d['numerical']
F_coh = {}
algorithm = algorithms['counterfactual_german']
constraints_embed = ['counterfactual_german']
objectives_embed = {}
model_master = em.model_selection(performance[performance['alg']==algorithm], constraints_embed, objectives_embed)
model_master['lb'] = 0.5
model_master['ub'] = None
model_master['SCM_counterfactuals'] = None
model_master['features'] = [[col for col in X_train.columns]]
model_master
y_ix_1 = np.where(y_train==1)
X1 = X_train.iloc[y_ix_1[0],:].copy().reset_index(drop=True, inplace=False)
u_index = 1
u = X_test_0.iloc[u_index,:]
print(u)
print('predicted label: %d' % (clf.predict([u])))
sp = True
mu = 0
tr_region = False
enlarge_tr = False
num_counterfactuals = 2
L = []
I = []
Pers_I = []
P = X_train.columns
F_b=[]
F_int=[]
data_pip=None
CEs, CEs_, final_model = ce_helpers.opt(X_train, X1, u, F_r, F_b, F_int, F_coh, I, L, Pers_I, P, 
                                        sp, mu, tr_region, enlarge_tr, num_counterfactuals, model_master, data_pip)
df_1 = ce_helpers.visualise_changes(clf, d, F_coh=F_coh, method = 'CE-OCL', CEs=CEs, CEs_ = CEs_, only_changes=True)

#Please set the depth level of decision tree you want to plot (set -1 for unlimited),
#and the minimum number of samples per node (set -1 for unlimited)
depth=5
minspn=5

#Fitting the Decision Tree model to the data with additional variables
if depth>0 and minspn>0:
  clfAV = DecisionTreeClassifier(min_samples_leaf=minspn, max_depth=depth, random_state=42)
elif depth>0:
  clfAV = DecisionTreeClassifier(max_depth=depth, random_state=42)
elif minspn>0:
  clfAV = DecisionTreeClassifier(min_samples_leaf=minspn, random_state=42)
else:
  clfAV = DecisionTreeClassifier(random_state=42)
X_train = train_data[train_data.columns[:-1]]
y_train = train_data[out]
X_test = test_data[test_data.columns[:-1]]
y_test = test_data[out]
clfAV.fit(X_train, y_train)
clfAV.predict(X_test.values)
R2.append(f1_score(y_test,PRED[len(PRED)-1],average='macro'))

#Plotting Decision Tree model (with additional variables) results
viz_model = dtreeviz.model(clfAV, X_train=X_train, y_train=y_train, class_names=class_names,
                           feature_names=[col for col in data.columns if col!=out])
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15) 
v.show()
