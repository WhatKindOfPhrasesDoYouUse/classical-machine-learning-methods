import os
import pandas as pd
import kagglehub as kh
from pandas import DataFrame
from helpers.validationHelper import validate_null_object

class TitanicLoader:
    """ Класс загрузки данных титаника с Kaggle. """

    def __init__(self):
        """Инициализирует DF массив null'ом. """

        self.df = None

    def load_from_kaggle(self) -> DataFrame:
        """ Загружает данные о титанике с Kaggle. """

        path_to_dataset_folder = kh.dataset_download("yasserh/titanic-dataset")
        csv_path = os.path.join(path_to_dataset_folder, "Titanic-Dataset.csv")
        self.df = pd.read_csv(csv_path)

        validate_null_object(
            self.df,
            f"Массив {type(self.df).__name__} является null")

        return self.df

    def get_data(self) -> DataFrame:
        """ Возвращает данные о титинике. """

        validate_null_object(
            self.df,
            f"Массив {type(self.df).__name__} является null")

        return self.df

    def get_info(self) -> str:
        """ Формирует и возвращает информацию о полученных данных. """

        validate_null_object(
            self.df,
            f"Массив {type(self.df).__name__} является null")

        return (f"Размер данных: {self.df.shape}\n"
                f"Колонки: {list(self.df.columns)}\n"
                f"Пропуски:\n{self.df.isnull().sum()}")