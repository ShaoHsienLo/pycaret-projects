from pycaret.classification import load_model
import os
import pandas as pd


loaded_model = load_model("best_recall_model_XGBClassifier")

data = pd.read_csv(os.path.join(r'C:\Users\samuello\Downloads\III\旺欉\code\labeled-data\1st-2st-labeling',
                                'all-labeled-data.csv'))
predict_data = data.drop(columns=['final label']).iloc[:1]
print(loaded_model.predict(predict_data))
