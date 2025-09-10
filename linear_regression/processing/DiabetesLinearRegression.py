import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from helpers import validationHelper

class DiabetesLinearRegression:
	""" Учитель и оценщик модели линейной регрессии. """

	def __init__(self):
		"Инициализирует экземпляр DiabetesLinearRegression. "

		self.model = LinearRegression()
		self.metrics = {}

	def train(self, X_train, y_train) -> None:
		""" Обучает модель линейной регрессии. """

		validationHelper.validate_null_array_element([
			X_train,
			y_train])

		self.model.fit(X_train, y_train)

	def evaluate(self, X_train, y_train, X_test, y_test) -> tuple:
		""" Рассчитывает метрики MSE и R^2 для обучающей и тестовой выборки. """

		y_train_pred = self.model.predict(X_train)
		y_test_pred = self.model.predict(X_test)

		validationHelper.validate_null_array_element([
			X_train,
			y_train,
			X_test,
			y_test,
			y_train_pred,
			y_test_pred
		])

		self.metrics = {
			"train": {
				"MSE": mean_squared_error(y_train, y_train_pred),
				"R2": r2_score(y_train, y_train_pred),
			},
			"test": {
				"MSE": mean_squared_error(y_test, y_test_pred),
				"R2": r2_score(y_test, y_test_pred),
			}
		}

		return self.metrics, y_train_pred, y_test_pred

	@staticmethod
	def gradient_step(x, y, w, b, lr = 0.01):
		""" Шаг градиентного спуска. """
		y_pred = np.dot(x, w) + b
		error = y_pred - y

		grad_w = 2 * error * x
		grad_b = 2 * error

		w_new = w - lr * grad_w
		b_new = b - lr * grad_b

		return w_new, b_new