from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from helpers import dataHelper, validationHelper

class DatasetSpliter:
	"""
	Разделитель датасета на обучающую и тестовую выборку.
	 	test_size - Доля тестовой выборки.
	 	random_state - Seed для воспроизводимости.
	"""

	default_test_size = 0.2
	default_random_state = 42

	def __init__(self, test_size=None, random_state=None):
		""" Инициализирует экземпляр типа DatasetSpliter. """

		self.test_size = dataHelper.set_default_if_input_value_null(
			default_value=self.default_test_size,
			input_value=test_size,
		)

		self.random_state = dataHelper.set_default_if_input_value_null(
			default_value=self.default_random_state,
			input_value=random_state,
		)

		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

		self.split_info = {}

	def split_dataset(self, X, y) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
		""" Разделяет данные на обучающую и тестовую выборки. """

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size=self.test_size, random_state=self.random_state
		)

		validationHelper.validate_null_array_element([
			self.X_train,
			self.X_test,
			self.y_train,
			self.y_test
		])

		return {
			'train': (self.X_train, self.y_train),
			'test': (self.X_test, self.y_test)
		}

	def get_split_dataset_info(self) -> str:
		""" Возвращает информацию о разделении на выборки. """

		return (f"X_train: {self.X_train.shape}\n"
				f"X_test: {self.X_test.shape}\n"
				f"y_train: {self.y_train.shape}\n"
				f"y_test: {self.y_test.shape}\n")


