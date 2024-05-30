import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
import lime
import lime.lime_tabular
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

"""
selected_columns = [0, 2, 3, 5, 10, 12, 13, 15, 16, 17, 18, 19, 22, 24, 25, 30, 32, 33, 36]
selected_columns = [0, 3, 12, 13, 15, 16, 17, 18, 19, 22, 24, 25, 30, 32, 33, 36]
selected_columns = [0, 2, 3, 10, 13, 15, 17, 18, 22, 24, 25, 30, 33, 36]
selected_columns = [0, 2, 3, 13, 15, 17, 18, 22, 24, 25, 33, 36]
selected_columns = [5, 6, 12, 15, 18, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 36]

selected_columns = [15, 23, 19, 5, 17, 21, 34, 16, 12, 31, 32, 24, 18, 22, 20, 33, 3, 36]
selected_columns = [15, 23, 19, 5, 17, 21, 34, 16, 12, 31, 36]
selected_columns = [15, 23, 19, 17, 16, 12, 32, 24, 20, 3, 36]

selected_columns = [5, 6, 9, 12, 15, 16, 17, 20, 23, 31, 32, 35, 36]
data = data.iloc[:, selected_columns]
"""
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
gpr= GaussianProcessClassifier(random_state=42)
xgb= XGBClassifier(random_state=42)
linr = LogisticRegression()
annad= MLPClassifier(max_iter=5000,hidden_layer_sizes=1000, random_state=42)
annlb= MLPClassifier(max_iter=5000,hidden_layer_sizes=1000,solver='lbfgs', random_state=42)
svrlin= SVC(kernel='linear', random_state=42)
svrpol= SVC(kernel='poly', random_state=42)
svrrbf01= SVC(kernel='rbf',C=0.1, random_state=42)
svrrbf1= SVC(kernel='rbf',C=1, random_state=42)
svrrbf10= SVC(kernel='rbf',C=10, random_state=42)
svrrbf100= SVC(kernel='rbf',C=100, random_state=42)
knn1= KNeighborsClassifier(n_neighbors=1)
knn3= KNeighborsClassifier(n_neighbors=3)
knn5= KNeighborsClassifier(n_neighbors=5)
knn7= KNeighborsClassifier(n_neighbors=7)
rfr2= RandomForestClassifier(max_depth=2, random_state=42)
rfr3= RandomForestClassifier(max_depth=3, random_state=42)
rfr4= RandomForestClassifier(max_depth=4, random_state=42)
rfr5= RandomForestClassifier(max_depth=5, random_state=42)
rfr10= RandomForestClassifier(max_depth=10, random_state=42)
rfr= RandomForestClassifier(random_state=42)

MODELS.extend([clfPJ2, clfPJ3, clfPJ4, clfPJ5, clfPJ6, clfPJ7, clfPJ8, clfPJ9, clfPJ10, clfPJ, clfPJ2_min5, clfPJ3_min5, clfPJ4_min5, clfPJ5_min5, 
                    clfPJ6_min5, clfPJ7_min5, clfPJ8_min5, clfPJ9_min5, clfPJ10_min5, clfPJ_min5, gbr, gpr, xgb, linr, annad, annlb, svrlin, svrpol, 
                    svrrbf01, svrrbf1, svrrbf10, svrrbf100, knn1, knn3, knn5, knn7, rfr2, rfr3, rfr4, rfr5, rfr10, rfr])

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
                           feature_names=['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility','P-Adaptability','P-Innovation','P-Reputation','P-Professionalism','P-Client Convenience','P-Client Service','P-Honesty','P-Integrity','P-Improvement','P-Self Directed','P-Initiative','P-Result','P-Responsibility','P-Performance','O-Team-oriented','O-Infosharing','O-Supportive','O-Flexibility','O-Adaptability','O-Innovation','O-Reputation','O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty','O-Integrity','O-Improvement','O-Self Directed','O-Initiative','O-Result','O-Responsibility','O-Performance','Tenure'],
                           class_names=['FIT','MISFIT'])
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15)     # render as SVG into internal object 
v.show()                 # pop up window

explainer = shap.Explainer(gbr.predict, X_test, random_state=42)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[41])

shap.summary_plot(shap_values, X_test, color="grayscale", feature_names=['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility','P-Adaptability','P-Innovation','P-Reputation','P-Professionalism','P-Client Convenience','P-Client Service','P-Honesty','P-Integrity','P-Improvement','P-Self Directed','P-Initiative','P-Result','P-Responsibility','P-Performance','O-Team-oriented','O-Infosharing','O-Supportive','O-Flexibility','O-Adaptability','O-Innovation','O-Reputation','O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty','O-Integrity','O-Improvement','O-Self Directed','O-Initiative','O-Result','O-Responsibility','O-Performance'])

clfPJT5_min5 = DecisionTreeClassifier(min_samples_leaf=5, max_depth=5)
X_train = train_data[train_data.columns[:-1]]
y_train = train_data[out]
X_test = test_data[test_data.columns[:-1]]
y_test = test_data[out]
clfPJT5_min5.fit(X_train, y_train)
clfPJT5_min5.predict(X_test.values)
R2.append(f1_score(y_test,PRED[36],average='macro'))

import dtreeviz
viz_model = dtreeviz.model(clfPJT5_min5,
                           X_train=X_train, y_train=y_train,
                           feature_names=['P-Team-oriented','P-Infosharing','P-Supportive','P-Flexibility','P-Adaptability','P-Innovation','P-Reputation','P-Professionalism','P-Client Convenience','P-Client Service','P-Honesty','P-Integrity','P-Improvement','P-Self Directed','P-Initiative','P-Result','P-Responsibility','P-Performance','O-Team-oriented','O-Infosharing','O-Supportive','O-Flexibility','O-Adaptability','O-Innovation','O-Reputation','O-Professionalism','O-Client Convenience','O-Client Service','O-Honesty','O-Integrity','O-Improvement','O-Self Directed','O-Initiative','O-Result','O-Responsibility','O-Performance','Tenure'],
                           class_names=['FIT','MISFIT'])
v = viz_model.view(label_fontsize=20, ticks_fontsize=12, title_fontsize=15)     # render as SVG into internal object 
v.show()                 # pop up window

