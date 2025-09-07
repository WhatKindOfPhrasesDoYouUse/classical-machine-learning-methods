from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_preprocessing.helpers.dataHelper import set_default_if_input_value_null

def create_categorical_transformer() -> Pipeline:
    """ Создает пайплайн для обработки категориальных признаков. """

    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

def create_numeric_transformer() -> Pipeline:
    """ Создает пайплайн для обработки числовых признаков. """

    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

class PipelineCreator:
    """
    Создатель конвейеров обработки данных и обучения модели.

        default_numeric_features - Числовые признаки по умолчанию.
        default_categorical_features - Категориальные признаки по умолчанию.
        default_random_state - Seed для воспроизводимости результатов.
        default_n_estimators - Количество деревьев в RandomForest по умолчанию.
        default_classifier - Классификатор по умолчанию.
    """

    default_numeric_features = ['Age', 'Fare']
    default_categorical_features = ['Pclass', 'Sex', 'Embarked']
    default_random_state = 42
    default_n_estimators = 100
    default_classifier = RandomForestClassifier(
        random_state=default_random_state,
        n_estimators=default_n_estimators)

    def __init__(self, numeric_features, categorical_features, classifier) -> None:
        """ Инициализирует создателя конвейера. """

        self.pipeline = None

        self.numeric_features = set_default_if_input_value_null(
            default_value=self.default_numeric_features,
            input_value=numeric_features
        )

        self.categorical_features = set_default_if_input_value_null(
            default_value=self.default_categorical_features,
            input_value=categorical_features
        )

        self.classifier = set_default_if_input_value_null(
            default_value=self.default_classifier,
            input_value=classifier
        )

    def create_preprocessor(self) -> ColumnTransformer:
        """ Создает ColumnTransformer для предобработки данных. """

        return ColumnTransformer(
            transformers=[
                ('numeric_features', create_numeric_transformer(), self.numeric_features),
                ('categorical_features', create_categorical_transformer(), self.categorical_features),
            ],
            remainder='drop'
        )

    def create_pipeline(self) -> Pipeline:
        """ Создает пайплайн. """

        preprocessor = self.create_preprocessor()

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.classifier),
        ])

        return self.pipeline

    def fit(self, X, y) -> Pipeline:
        """Обучает пайплайн на обучающей выборке."""

        if self.pipeline is None:
            self.create_pipeline()

        self.pipeline.fit(X, y)

        return self.pipeline