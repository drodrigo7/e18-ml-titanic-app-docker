# main.py
# ==================================================
# settings
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------
# standard
import pickle
# requirements
import typer
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# --------------------------------------------------

app = typer.Typer()

@app.command()
def model() -> None:
    '''This command trains and outputs the machine learning models.
    '''
    
    # [NOTE] datasets
    # --------------------------------------------------
    df_train = pd.read_csv('./data/train.csv', encoding='utf-8')
    df_test = pd.read_csv('./data/test.csv', encoding='utf-8')
    
    df_train['T'], df_test['T'] = 1, 2
    df = df_train.append(df_test, ignore_index=True)
    
    # [NOTE] features for the example
    # --------------------------------------------------
    keeps = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'T']
    df = df[keeps]
    
    # [NOTE] features and target / data split
    # --------------------------------------------------
    X = df[df['T']==1].drop(columns=['PassengerId', 'Survived', 'T'])
    y = df[df['T']==1]['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # [NOTE] pipeline
    # --------------------------------------------------
    mapper = DataFrameMapper([
        (['Age', 'Fare'], [SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler()]),
        (['SibSp', 'Parch'], None),
        (['Pclass'], OneHotEncoder()),
        (['Sex'], LabelEncoder()),
        (['Embarked'], [SimpleImputer(missing_values=np.nan, strategy='most_frequent'), LabelBinarizer()])
    ])
    pre_steps = [('features', mapper), ('selection', PCA(n_components=.95))]
    
    # [NOTE] models
    # --------------------------------------------------
    models = {
        'XGBC': [
            ('XGBC', XGBClassifier()), dict(XGBC__max_depth=[4, 8, 12], XGBC__n_estimators=[10, 100, 500])
        ],
        'LGBM': [
            ('LGBM', LGBMClassifier()), dict(LGBM__max_depth=[4, 8, 12], LGBM__n_estimators=[50, 100, 500])
        ]
    }
    
    # [NOTE] evaluation
    # --------------------------------------------------
    results = dict()
    for name, items in models.items():
        logger.info('Modelo {} en proceso...'.format(name))
        
        steps = pre_steps + [items[0]]
        pipeline = Pipeline(steps)
        model = GridSearchCV(pipeline, param_grid=items[1], cv=3)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results[name] = {
            'scores': (round(accuracy_score(y_test, y_pred), 6), round(roc_auc_score(y_test, y_pred), 6)),
            'best_params': model.best_params_
        }
        
        with open(f'./model/{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        logger.info('Modelo {} terminado'.format(name))
    
    logger.info('Fin de entrenamiento y predicciones')
    logger.info(results)

@app.command()
def test() -> None:
    '''This command uses one output model from the training step and
    tests its usage with sample input.
    '''
    
    with open('./model/XGBC_model.pkl', 'rb') as bm:
        best_model = pickle.load(bm)
    
    sample_data = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25,
        'SibSp': 1,
        'Parch': 1,
        'Fare': 15,
        'Embarked': 'S'
    }
    row = pd.DataFrame(sample_data, index=[0])
    result = best_model.predict(row)[0]
    logger.info('Test result: {}'.format(result))

if __name__ == '__main__':
    app()
