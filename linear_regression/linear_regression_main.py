from linear_regression.data.DiabetesDatasetLoader import DiabetesDatasetLoader
from linear_regression.processing.DiabetesLinearRegression import DiabetesLinearRegression
from linear_regression.visualization.RegressionVisualizer import RegressionVisualizer
from helpers.DatasetSpliter import DatasetSpliter

def main():
    dataset = DiabetesDatasetLoader()
    dataset.load_data()

    X = dataset.get_data()
    y = dataset.get_target()

    splitter = DatasetSpliter()
    split_data = splitter.split_dataset(X, y)
    X_train, y_train = split_data["train"]
    X_test, y_test = split_data["test"]

    print(splitter.get_split_dataset_info())

    regressor = DiabetesLinearRegression()
    regressor.train(X_train, y_train)
    metrics, y_train_pred, y_test_pred = regressor.evaluate(X_train, y_train, X_test, y_test)

    print("Результаты линейной регрессии:")
    print(f"Train -> MSE: {metrics['train']['MSE']:.2f}, R2: {metrics['train']['R2']:.4f}")
    print(f"Test  -> MSE: {metrics['test']['MSE']:.2f}, R2: {metrics['test']['R2']:.4f}")

    RegressionVisualizer.plot_predictions(y_test, y_test_pred)
    RegressionVisualizer.plot_residuals(y_test, y_test_pred)

if __name__ == '__main__':
	main()