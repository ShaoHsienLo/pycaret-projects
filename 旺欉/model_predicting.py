from pycaret.classification import load_model
import os
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def f(x):
    map_dict = {'ingot': '錠溫異常', 'oil_pressure': '油壓異常', 'mould': '模溫異常', 'bucket': '盛錠筒溫異常'}
    return map_dict[x]


loaded_model = load_model("best_recall_model_LGBMClassifier")

data = pd.read_csv(os.path.join(r'C:\Users\samuello\Downloads\III\常用\旺欉\code\labeled-data\1st-2st-labeling',
                                'all-labeled-data.csv'))
predict_data = data.drop(columns=['final label']).iloc[:1]

X_train = pd.read_csv("train.csv").drop(columns=['final label'])
X_test = pd.read_csv("test.csv").drop(columns=['final label'])
cols = X_train.columns

print(loaded_model.predict_proba(predict_data[cols]))

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['A', 'B'],
                                 mode='classification')
# i = random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test.iloc[0], loaded_model.predict_proba)

lst = exp.as_list()
target = lst[0][0].split('_')[0]
target = list(map(f, [target]))[0]
print(target)
