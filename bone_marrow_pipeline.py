import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.io import arff


def load_data():
    data_file = Path(__file__).with_name('bone-marrow.arff')
    if not data_file.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {data_file}\n"
            "Coloque bone-marrow.arff na mesma pasta deste script."
        )

    raw_data = arff.loadarff(data_file)
    df = pd.DataFrame(raw_data[0])
    if 'Disease' in df.columns:
        df.drop(columns=['Disease'], inplace=True)
    return df


def main():
    df = load_data()
    warnings.filterwarnings("ignore", category=UserWarning)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in df.columns[df.nunique() == 2]:
        df[c] = (df[c] == 1).astype(float)

    print('\n' + '='*70)
    print('DATASET SUMMARY'.center(70))
    print('='*70)
    print(f'Rows: {len(df)}')
    print(f'Columns: {df.shape[1]}')
    print('Unique values per column:')
    print(df.nunique().to_string())

    if 'survival_status' not in df.columns or 'survival_time' not in df.columns:
        raise KeyError("O dataset precisa conter as colunas 'survival_status' e 'survival_time'.")

    y = LabelEncoder().fit_transform(df['survival_status'].astype(str))
    X = df.drop(columns=['survival_time', 'survival_status'])

    num_cols = X.columns[X.nunique() > 7]
    cat_cols = X.columns[X.nunique() <= 7]
    missing_cols = X.columns[X.isnull().sum() > 0]

    print('\n' + '='*70)
    print('FEATURE SUMMARY'.center(70))
    print('='*70)
    print(f'Feature count: {X.shape[1]}')
    print(f'Numeric columns: {len(num_cols)}')
    print(f'Categorical columns: {len(cat_cols)}')
    print(f'Missing value columns: {len(missing_cols)}')
    if len(missing_cols):
        print(', '.join(missing_cols.tolist()))
    else:
        print('None')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2
    )

    cat_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'ohe',
                OneHotEncoder(
                    sparse_output=False,
                    drop='first',
                    handle_unknown='ignore',
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ('cat_process', cat_pipeline, cat_cols),
            ('num_process', num_pipeline, num_cols),
        ],
        remainder='drop',
    )

    pipeline = Pipeline(
        [
            ('preprocess', preprocess),
            ('pca', PCA()),
            ('clf', LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(x_train, y_train)
    train_score = pipeline.score(x_train, y_train)
    test_score = pipeline.score(x_test, y_test)

    print('\n' + '='*70)
    print('PIPELINE PERFORMANCE'.center(70))
    print('='*70)
    print(f'Train accuracy: {train_score:.4f}')
    print(f'Test accuracy:  {test_score:.4f}')

    max_pca = min(37, X.shape[1])
    if max_pca < 1:
        raise ValueError('Não há recursos suficientes para aplicar PCA.')

    search_space = [
        {
            'clf': [LogisticRegression(max_iter=1000)],
            'clf__C': np.logspace(-4, 2, 10),
            'pca__n_components': np.arange(1, min(10, max_pca) + 1),
        }
    ]

    gs = GridSearchCV(pipeline, search_space, cv=3, n_jobs=1)
    gs.fit(x_train, y_train)

    best_model = gs.best_estimator_
    best_params = best_model.named_steps['clf'].get_params()

    print('\n' + '='*70)
    print('BEST MODEL SUMMARY'.center(70))
    print('='*70)
    print(f'Model: {best_model.named_steps["clf"].__class__.__name__}')
    print(f'C: {best_params["C"]}')
    print(f'Penalty: {best_params["penalty"]}')
    print(f'Solver: {best_params["solver"]}')
    print(f'Max iter: {best_params["max_iter"]}')
    print(f'PCA components: {best_model.named_steps["pca"].n_components}')
    print(f'Best model test accuracy: {best_model.score(x_test, y_test):.4f}')


if __name__ == '__main__':
    main()
