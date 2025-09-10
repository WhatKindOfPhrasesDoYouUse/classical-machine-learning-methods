from typing import List, Any, Dict
import numpy as np
from sklearn.datasets import load_diabetes
from helpers import validationHelper

class DiabetesDatasetLoader:
	""" Класс загрузки данных. """

	def __init__(self):
		""" Инициализирует новый экземпляр DiabetesDatasetLoader. """

		self.data = None
		self.target = None
		self.feature_names = None
		self.dataset_info = {}

	def load_data(self) -> None:
		""" Загружает датасет diabetes из sklearn.datasets """

		dataset = load_diabetes()

		self.data = dataset.data
		self.target = dataset.target
		self.feature_names = dataset.feature_names

		self.dataset_info = dataset.dataset_info = {
			'n_samples': self.data.shape[0],
			'n_features': self.data.shape[1],
			'target_range': (float(self.target.min()), float(self.target.max())),
			'target_mean': float(self.target.mean()),
			'feature_names': self.feature_names,
			'description': dataset.DESCR[:200] + "..." if dataset.DESCR else "Описание отсутствует"
		}

	def get_data(self) -> np.ndarray:
		""" Возвращает матрицу признаков датасета. """

		validationHelper.validate_null_object(
			self.data,
			f"{type(self.data).__name__} является null")

		return self.data

	def get_target(self) -> np.ndarray:
		""" Возвращает вектор целевой переменной. """

		validationHelper.validate_null_object(
			self.target,
			f"{type(self.target).__name__} является null")

		return self.target

	def get_feature_names(self) -> List[str]:
		""" Возвращает массив с названиями признаков. """

		validationHelper.validate_null_object(
			self.feature_names,
			f"{type(self.feature_names).__name__} является null")

		return self.feature_names

	def get_dataset_info(self) -> Dict[str, Any]:
		""" Возвращает метаинформацию и датасете. """

		validationHelper.validate_null_object(
			self.dataset_info,
			f"{type(self.dataset_info).__name__} является null")

		return self.dataset_info