# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
simplefilter(action='ignore', category=FutureWarning)

url = 'weatherAUS.csv'
data = pd.read_csv(url)

    
data.MinTemp.replace(np.nan, 12, inplace = True)
data.MaxTemp.replace(np.nan, 23, inplace = True)
data.Rainfall.replace(np.nan, 2, inplace = True)
data.WindGustSpeed.replace(np.nan, 40, inplace = True)
data.WindSpeed9am.replace(np.nan, 14, inplace = True)
data.WindSpeed3pm.replace(np.nan, 19, inplace = True)
data.Humidity9am.replace(np.nan, 69, inplace = True)
data.Humidity3pm.replace(np.nan, 51, inplace = True)
data.Pressure9am.replace(np.nan, 1018, inplace = True)
data.Pressure3pm.replace(np.nan, 1015, inplace = True)
data.Temp9am.replace(np.nan, 17, inplace = True)
data.Temp3pm.replace(np.nan, 22, inplace = True)

data.RainTomorrow.replace(['Yes', 'No'], [1,0], inplace = True)
data.RainToday.replace(['Yes', 'No'], [1,0], inplace = True)
data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'WindDir9am', 'WindDir3pm', 'WindGustDir', 'WindDir3pm' ], axis = 1, inplace = True)
data.dropna(axis=0, how='any', inplace=True)
    
x = np.array(data.drop(['RainTomorrow'], 1))
y = np.array(data.RainTomorrow)  # 0 No llueve, 1 Si llueve

def metricas_entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


def matriz_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr
    
def show_metrics(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print('Matriz de Confusion:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')

#Punto 1 y #Punto2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model1 = LogisticRegression()
model1, acc_validation1, acc_test1, y_pred =metricas_entrenamiento(model1, x_train, x_test, y_train, y_test)
matriz_confusion1, AUC1, fpr1, tpr1  = matriz_confusion_auc(model1, x_test, y_test, y_pred)
show_metrics('Linear Regression', AUC1, acc_validation1, acc_test1, y_test, y_pred)
classification1 = classification_report(y_test, y_pred, output_dict=True)
clas_model1 = pd.DataFrame(classification1).transpose()

model2 = DecisionTreeClassifier()
model2, acc_validation2, acc_test2, y_pred =metricas_entrenamiento(model2, x_train, x_test, y_train, y_test)
matriz_confusion2, AUC2, fpr2, tpr2  = matriz_confusion_auc(model2, x_test, y_test, y_pred)
show_metrics('Decision Tree Classifier',AUC2, acc_validation2, acc_test2, y_test, y_pred)
classification2 = classification_report(y_test, y_pred, output_dict=True)
clas_model2 = pd.DataFrame(classification2).transpose()

model3 = KNeighborsClassifier(n_neighbors = 3)
model3, acc_validation3, acc_test3, y_pred =metricas_entrenamiento(model3, x_train, x_test, y_train, y_test)
matriz_confusion3, AUC3, fpr3, tpr3  = matriz_confusion_auc(model3, x_test, y_test, y_pred)
show_metrics('KNeighborns Classifier',AUC3, acc_validation3, acc_test3, y_test, y_pred)
classification3 = classification_report(y_test, y_pred, output_dict=True)
clas_model3 = pd.DataFrame(classification3).transpose()

model4 = GaussianNB()
model4, acc_validation4, acc_test4, y_pred =metricas_entrenamiento(model4, x_train, x_test, y_train, y_test)
matriz_confusion4, AUC4, fpr4, tpr4  = matriz_confusion_auc(model4, x_test, y_test, y_pred)
show_metrics('GaussianNB',AUC4, acc_validation4, acc_test4, y_test, y_pred)
classification4 = classification_report(y_test, y_pred, output_dict=True)
clas_model4 = pd.DataFrame(classification4).transpose()

model5 = RandomForestClassifier()
model5, acc_validation5, acc_test5, y_pred =metricas_entrenamiento(model5, x_train, x_test, y_train, y_test)
matriz_confusion5, AUC5, fpr5, tpr5  = matriz_confusion_auc(model5, x_test, y_test, y_pred)
show_metrics('RandomForestClassifier',AUC5, acc_validation5, acc_test5, y_test, y_pred)
classification5 = classification_report(y_test, y_pred, output_dict=True)
clas_model5 = pd.DataFrame(classification5).transpose()

#Punto 3
precision_model1 = clas_model1.iat[3,0]
precision_model2 = clas_model2.iat[3,0]
precision_model3 = clas_model3.iat[3,0]
precision_model4 = clas_model4.iat[3,0]
precision_model5 = clas_model5.iat[3,0]
recall_model1 = clas_model1.iat[3,1]
recall_model2 = clas_model2.iat[3,1]
recall_model3 = clas_model3.iat[3,1]
recall_model4 = clas_model4.iat[3,1]
recall_model5 = clas_model5.iat[3,1]
f1score_model1 = clas_model1.iat[3,2]
f1score_model2 = clas_model2.iat[3,2]
f1score_model3 = clas_model3.iat[3,2]
f1score_model4 = clas_model4.iat[3,2]
f1score_model5 = clas_model5.iat[3,2]

print('Resultados')
modelos = {'Acurracy de Validacion': [acc_validation1, acc_validation2, acc_validation3, acc_validation4, acc_validation5],
           'Acurracy de Test': [acc_test1, acc_test2, acc_test3, acc_test4, acc_test5],
           'Presicion': [precision_model1,precision_model2, precision_model3, precision_model4, precision_model5],
           'Recall': [recall_model1, recall_model2, recall_model3, recall_model4, recall_model5],
           'f1-score': [f1score_model1, f1score_model2, f1score_model3, f1score_model4, f1score_model5],
           'AUC': [AUC1, AUC2, AUC3, AUC4, AUC5]
           }

tabla = pd.DataFrame(modelos, columns = ['Acurracy de Validacion', 'Acurracy de Test', 'AUC', 'Presicion','Recall', 'f1-score'], index =['LogisticRegression', 'DecissionTreeClassifier', 'KNheighorsClassifier', 'GaussianNB', 'RandomForestClassifier'])

print (tabla)
#Punto 5 

fig, (ax,ax2, ax3, ax4, ax5) = plt.subplots(ncols = 5)
fig.tight_layout()
fig.subplots_adjust(wspace = 0.3, hspace = 0.3, right = 3, bottom = 0.4, top = 1)
a = sns.heatmap(matriz_confusion1, ax=ax)
b = sns.heatmap(matriz_confusion2, ax=ax2)
c = sns.heatmap(matriz_confusion3, ax=ax3)
d = sns.heatmap(matriz_confusion4, ax=ax4)
e = sns.heatmap(matriz_confusion5, ax=ax5)
a.set_title('Matriz de confunsion LR')
b.set_title('Matriz de confunsion DTC')
c.set_title('Matriz de confunsion KNC')
d.set_title('Matriz de confunsion GNB')
e.set_title('Matriz de confunsion RFC')

#Punto 7

plt.show()
plt.plot(fpr1, tpr1, color='black', label='ROC1')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.plot(fpr2, tpr2, color='yellow', label='ROC2')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.plot(fpr3, tpr3, color='green', label='ROC3')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.plot(fpr4, tpr4, color='purple', label='ROC4')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.plot(fpr5, tpr5, color='orange', label='ROC5')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

plt.show()
