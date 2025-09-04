import os
import numpy as np
import pandas as pd
import kagglehub as kh
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

""" 
    Загружает и возвращает данные c Kaggle. 
"""
def load_data():
    path_to_dataset_folder = kh.dataset_download("yasserh/titanic-dataset")
    csv_path = os.path.join(path_to_dataset_folder, "Titanic-Dataset.csv")
    df = pd.read_csv(csv_path)
    return df

""" 
    Выделяет признаки и добавляет пропуски в поле возраста. 
"""
def highlights_signs():
    df = load_data()
    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    df = df[cols].copy() # Оставляет данные только из выбранных колонок.
    df.loc[df.sample(frac=0.1, random_state=42).index, 'Age'] = np.nan
    #print(f"Количество пропусков: \n {df.isna().sum()}")
    return df

""" 
    Разделяет данные на обучающую и тестовую выборку 
    в соотношении 80/20. 
"""
def splid_data():
    df = highlights_signs()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Обучающая выборка: {X_train.shape} {y_train.shape}")
    print(f"Тестовая выборка: {X_test.shape} {y_test.shape}")

    return X_train, X_test, y_train, y_test


def main():
    split = splid_data()

if __name__ == '__main__':
    main()