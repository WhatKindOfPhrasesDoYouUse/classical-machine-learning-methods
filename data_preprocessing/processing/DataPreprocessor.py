import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from data_preprocessing.helpers.validationHelper import validate_null_object
from data_preprocessing.helpers.dataHelper import set_default_if_input_value_null

class DataPreprocessor:
    """
    Предварительный обработчик датасета перед машинным обучением.
        default_features - Признаки по умолчанию для выборки из датасета.
        default_fraction - Доля пропущенных значений по умолчанию.
        default_random_state - Seed для воспроизводимости случайных операций.
        default_test_size - Размер тестовой выборки по умолчанию.
        missing_column - Колонка для добавления пропущенных значений.
    """

    default_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    default_fraction = 0.1
    default_random_state = 42
    default_test_size = 0.2
    missing_column = "Age"

    def __init__(self, df):
        """ Инициализирует объект DataPreprocessor. """

        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def select_features(self, features=None) -> DataFrame:
        """ Выбирает указанные в массиве features признаки из датасета. """

        features = set_default_if_input_value_null(
            default_value=self.default_features,
            input_value=features
        )

        self.df = self.df[features]

        validate_null_object(
            obj=self.df,
            message=f"Признаки {type(self.df).__name__} отсутствуют в датасете."
        )

        return self.df

    def add_missing_values(self, fraction=None, random_state=None):
        """ Добавляет пропуски в поле Age. """

        fraction = set_default_if_input_value_null(
            default_value=self.default_fraction,
            input_value=fraction
        )

        random_state = set_default_if_input_value_null(
            default_value=self.default_random_state,
            input_value=random_state
        )

        self.df.loc[
            self.df.sample(frac=fraction, random_state=random_state).index,
            self.missing_column
        ] = np.nan

        return self.df

    def split_data(self, test_size=None, random_state=None):
        """ Делит датасет на обучающую и тестовую выборку. """

        test_size = set_default_if_input_value_null(
            default_value=self.default_test_size,
            input_value=test_size
        )

        random_state = set_default_if_input_value_null(
            default_value=self.default_random_state,
            input_value=random_state
        )

        X = self.df.drop("Survived", axis=1)
        y = self.df["Survived"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_split_info(self) -> str:
        """ Возвращает значения обучающей и тестовой выборки. """

        validate_null_object(
            self.X_train,
            f"Обучающие признаки {type(self.X_train).__name__} не инициализированы")

        validate_null_object(
            self.X_test,
            f"Тестовые признаки {type(self.X_test).__name__} не инициализированы")

        validate_null_object(
            self.y_train,
            f"Обучающие целевые значения {type(self.y_train).__name__} не инициализированы")

        validate_null_object(
            self.y_test,
            f"Тестовые целевые значения {type(self.y_test).__name__} не инициализированы")

        return (f"X_train: {self.X_train.shape}\n"
                f"X_test: {self.X_test.shape}\n"
                f"y_train: {self.y_train.shape}\n"
                f"y_test: {self.y_test.shape}\n")