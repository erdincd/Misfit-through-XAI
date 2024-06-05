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

random.seed(42)
np.random.seed(42)
random_state=check_random_state(42)

raw_data= pd.read_csv("Data.csv")
data_inp=raw_data[raw_data.columns[0:37]]
data_inp.columns=['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility','P-Adaptability','P-Innovation',
                  'P-Reputation','P-Professionalism','P-Client Convenience','P-Client Service','P-Honesty','P-Integrity',
                  'P-Improvement','P-Self Directed','P-Initiative','P-Result','P-Responsibility','P-Performance',
                  'O-Team-oriented','O-Infosharing','O-Supportive','O-Flexibility','O-Adaptability','O-Innovation',
                  'O-Reputation','O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty','O-Integrity',
                  'O-Improvement','O-Self Directed','O-Initiative','O-Result','O-Responsibility','O-Performance','Tenure']

out="PO"

data=data_inp.copy()
data.insert(37, out, raw_data[out])
data=data.dropna(axis=0, how='any', subset=None, inplace=False)

predicted_data = data.copy()

R2=[]
PRED=[]
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

test_data=data.iloc[[5, 9, 16, 20, 23, 24, 35, 42, 46, 47, 56, 59, 61, 68, 70, 72, 73, 78, 82, 85, 91, 98,
    110, 115, 122, 126, 130, 132, 134, 141, 143, 149, 158, 170, 171, 177, 182, 185, 190, 192,
    198, 202, 208, 213, 219, 220, 223, 230, 239, 240, 247]]

train_data=data.iloc[[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 28, 
                      29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 48, 49, 50, 51, 52, 53, 
                      54, 55, 57, 58, 60, 62, 63, 64, 65, 66, 67, 69, 71, 74, 75, 76, 77, 79, 80, 81, 83, 
                      84, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 
                      107, 108, 109, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 123, 124, 125, 127, 
                      128, 129, 131, 133, 135, 136, 137, 138, 139, 140, 142, 144, 145, 146, 147, 148, 150, 
                      151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 
                      169, 172, 173, 174, 175, 176, 178, 179, 180, 181, 183, 184, 186, 187, 188, 189, 191, 
                      193, 194, 195, 196, 197, 199, 200, 201, 203, 204, 205, 206, 207, 209, 210, 211, 212, 
                      214, 215, 216, 217, 218, 221, 222, 224, 225, 226, 227, 228, 229, 231, 232, 233, 234, 
                      235, 236, 237, 238, 241, 242, 243, 244, 245, 246, 248]]

X_train = train_data[train_data.columns[:-2]]
y_train = train_data[out]
X_test = test_data[test_data.columns[:-2]]
y_test = test_data[out]

for i in range(len(MODELS)):
      if i>31 & i<36:
        MODELS[i].fit(X_train, y_train)
        PRED.append(MODELS[i].predict(X_test.values))
        R2.append(f1_score(y_test,PRED[i],average='macro'))     
      else:
        MODELS[i].fit(X_train, y_train)
        PRED.append(MODELS[i].predict(X_test))
        R2.append(f1_score(y_test,PRED[i],average='macro'))

        import matplotlib.pyplot as plt
import numpy as np

def tornado_plot(attributes, values1, values2, title='Tornado Plot'):
    # Calculate the absolute sum of values for each attribute
    scores = [abs(v1 + v2) for v1, v2 in zip(values1, values2)]

    # Sort attributes based on scores
    sorted_indices = np.argsort(scores)[::-1]
    attributes = [attributes[i] for i in sorted_indices]
    values1 = [values1[i] for i in sorted_indices]
    values2 = [values2[i] for i in sorted_indices]

    fig, ax = plt.subplots()
    # Calculate the position for the sorted attributes
    attribute_positions = np.arange(len(attributes))

    # Create bars for values1 (e.g., positive values)
    ax.barh(attribute_positions - 0.2, values1, height=0.4, color='dimgrey', label='P')

    # Create bars for values2 (e.g., negative values)
    ax.barh(attribute_positions + 0.2, values2, height=0.4, color='lightgray', label='O')

    # Set labels at the sorted attribute positions
    ax.set_yticks(attribute_positions)
    ax.set_yticklabels(attributes)
    ax.invert_yaxis()  # Invert the y-axis to have the labels at the center

    # Draw horizontal lines between attributes
    for i in range(len(attributes) - 1):
        ax.axhline(attribute_positions[i] + 0.5, color='gray', linestyle='dashed', linewidth=1)

    # Set labels and title
    ax.set_xlabel('Coefficients')
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

labels=['Team-oriented','Infosharing','Supportive','Flexibility',
                  'Adaptability','Innovation','Reputation','Professionalism','Client Convenience',
                  'Client Service','Honesty','Integrity','Improvement','Self Directed',
                  'Initiative','Result','Responsibility','Performance']

all=linr.coef_
p=all[0]
pos=p[0:18]
neg=p[18:37]
tornado_plot(labels, pos, neg, title="Logistic Regression Model")

import dtreeviz
viz_model = dtreeviz.model(clfPJ4_min5,
                           X_train=X_train, y_train=y_train,
                           feature_names=['P-Team-oriented','P-Infosharing','P-Supportive',
                                          'P-Flexibility','P-Adaptability','P-Innovation','P-Reputation',
                                          'P-Professionalism','P-Client Convenience','P-Client Service',
                                          'P-Honesty','P-Integrity','P-Improvement','P-Self Directed',
                                          'P-Initiative','P-Result','P-Responsibility','P-Performance',
                                          'O-Team-oriented','O-Infosharing','O-Supportive',
                                          'O-Flexibility','O-Adaptability','O-Innovation','O-Reputation',
                                          'O-Professionalism','O-Client Convenience','O-Client Service',
                                          'O-Honesty','O-Integrity','O-Improvement','O-Self Directed',
                                          'O-Initiative','O-Result','O-Responsibility','O-Performance','Tenure'],
                           class_names=['FIT','MISFIT'])
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15)     # render as SVG into internal object 
v.show()                 # pop up window

explainer = shap.Explainer(gbr.predict, X_test, seed=123456789)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[41])

shap.summary_plot(shap_values, X_test, color="grayscale", 
                  feature_names=['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility',
                                 'P-Adaptability','P-Innovation','P-Reputation','P-Professionalism',
                                 'P-Client Convenience','P-Client Service','P-Honesty','P-Integrity',
                                 'P-Improvement','P-Self Directed','P-Initiative','P-Result',
                                 'P-Responsibility','P-Performance','O-Team-oriented','O-Infosharing',
                                 'O-Supportive','O-Flexibility','O-Adaptability','O-Innovation','O-Reputation',
                                 'O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty',
                                 'O-Integrity','O-Improvement','O-Self Directed','O-Initiative','O-Result',
                                 'O-Responsibility','O-Performance'])

y_train=1-y_train
y_test=1-y_test
d = {
    'target': 'PO',
    'numerical':['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility','P-Adaptability','P-Innovation',
                  'P-Reputation','P-Professionalism','P-Client Convenience','P-Client Service','P-Honesty','P-Integrity',
                  'P-Improvement','P-Self Directed','P-Initiative','P-Result','P-Responsibility','P-Performance',
                  'O-Team-oriented','O-Infosharing','O-Supportive','O-Flexibility','O-Adaptability','O-Innovation',
                  'O-Reputation','O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty','O-Integrity',
                  'O-Improvement','O-Self Directed','O-Initiative','O-Result','O-Responsibility','O-Performance']
}
version = 'Data_v1'
alg_list_cf = ['gbm']
#alg_list_dr = ['mlp', 'linear', 'svm', 'rf']
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
y_pred, y_pred_0, X_test_0, models = ce_helpers.load_model(algorithms, outcome_dict, 'counterfactual_german')  # it should be X_test instead of X
X_test_0.head()
clf = models['counterfactual_german']
F_r = d['numerical']
F_coh = {}
algorithm = algorithms['counterfactual_german']
constraints_embed = ['counterfactual_german']
objectives_embed = {}
model_master = em.model_selection(performance[performance['alg']==algorithm], constraints_embed, objectives_embed)
model_master['lb'] = 0.5  # this can be changed but it is generally equal to 0.5
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
# immutable features
I = []
# conditionally mutable features
Pers_I = []
P = X_train.columns
F_b=[]
F_int=[]
data_pip=None
CEs, CEs_, final_model = ce_helpers.opt(X_train, X1, u, F_r, F_b, F_int, F_coh, I, L, Pers_I, P, 
                                        sp, mu, tr_region, enlarge_tr, num_counterfactuals, model_master, data_pip)
df_1 = ce_helpers.visualise_changes(clf, d, F_coh=F_coh, method = 'CE-OCL', CEs=CEs, CEs_ = CEs_, only_changes=True)
df_1

clfPJT5_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=5, random_state=42)
X_train = train_data[train_data.columns[:-1]]
y_train = train_data[out]
X_test = test_data[test_data.columns[:-1]]
y_test = test_data[out]
clfPJT5_min5.fit(X_train, y_train)
clfPJT5_min5.predict(X_test.values)
R2.append(f1_score(y_test,PRED[21],average='macro'))

import dtreeviz
viz_model = dtreeviz.model(clfPJT5_min5,
                           X_train=X_train, y_train=y_train,
                           feature_names=['P-Team-oriented','P-Infosharing','P-Supportive',
                                          'P-Flexibility','P-Adaptability','P-Innovation','P-Reputation',
                                          'P-Professionalism','P-Client Convenience','P-Client Service',
                                          'P-Honesty','P-Integrity','P-Improvement','P-Self Directed',
                                          'P-Initiative','P-Result','P-Responsibility','P-Performance',
                                          'O-Team-oriented','O-Infosharing','O-Supportive',
                                          'O-Flexibility','O-Adaptability','O-Innovation','O-Reputation',
                                          'O-Professionalism','O-Client Convenience','O-Client Service',
                                          'O-Honesty','O-Integrity','O-Improvement','O-Self Directed',
                                          'O-Initiative','O-Result','O-Responsibility','O-Performance','Tenure'],
                           class_names=['FIT','MISFIT'])
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15)     # render as SVG into internal object 
v.show()                 # pop up window
