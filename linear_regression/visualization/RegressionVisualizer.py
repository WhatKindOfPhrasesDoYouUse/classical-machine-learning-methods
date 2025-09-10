import matplotlib.pyplot as plt

class RegressionVisualizer:
	""" Визуализатор регрессии на графике. """

	@staticmethod
	def plot_predictions(y_true, y_pred):
		""" График истинные vs предсказанные значения. """

		plt.figure(figsize=(6, 6))
		plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
		plt.plot([y_true.min(), y_true.max()],
				 [y_true.min(), y_true.max()],
				 color="red", linestyle="--", linewidth=2)
		plt.xlabel("Истинные значения")
		plt.ylabel("Предсказанные значения")
		plt.title("Истинные vs Предсказанные")
		plt.grid(True)
		plt.show()

	@staticmethod
	def plot_residuals(y_true, y_pred):
		""" График остатков (y_true - y_pred). """

		residuals = y_true - y_pred
		plt.figure(figsize=(6, 4))
		plt.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
		plt.axhline(0, color="red", linestyle="--", linewidth=2)
		plt.xlabel("Предсказанные значения")
		plt.ylabel("Остатки")
		plt.title("График остатков")
		plt.grid(True)
		plt.show()