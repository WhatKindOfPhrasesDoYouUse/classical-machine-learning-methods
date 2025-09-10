import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from helpers.dataHelper import set_default_if_input_value_null

class ManualPreprocessor:
	"""
	Ручной препроцессинг и обучение классификатора.

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


	def __init__(self, numeric_features=None, categorical_features=None, classifier=None) -> None:
		""" Инициализирует ручной препроцессинг. """

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

	def process_and_train(self, X_train, y_train, X_test, y_test):
		""" Выполняет полный цикл препроцессинга и обучения модели. """

		num_imputer = SimpleImputer(strategy='median')
		cat_imputer = SimpleImputer(strategy='most_frequent')
		scaler = StandardScaler()
		encoder = OneHotEncoder(handle_unknown='ignore')

		x_train_num_imputed = num_imputer.fit_transform(X_train[self.numeric_features])
		x_train_cat_imputed = cat_imputer.fit_transform(X_train[self.categorical_features])

		x_train_num_scaled = scaler.fit_transform(x_train_num_imputed)
		x_train_cat_encoded = encoder.fit_transform(x_train_cat_imputed).toarray()

		x_train_processed = np.hstack([x_train_num_scaled, x_train_cat_encoded])

		x_test_num_imputed = num_imputer.transform(X_test[self.numeric_features])
		x_test_cat_imputed = cat_imputer.transform(X_test[self.categorical_features])

		x_test_num_scaled = scaler.transform(x_test_num_imputed)
		x_test_cat_encoded = encoder.transform(x_test_cat_imputed).toarray()

		x_test_processed = np.hstack([x_test_num_scaled, x_test_cat_encoded])

		self.classifier.fit(x_train_processed, y_train)
		y_pred = self.classifier.predict(x_test_processed)

		return accuracy_score(y_test, y_pred)