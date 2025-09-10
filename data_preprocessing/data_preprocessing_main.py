import time
from collections.abc import Callable
from typing import Any
from sklearn.metrics import accuracy_score
from data_preprocessing.data.TitanicLoader import TitanicLoader
from data_preprocessing.processing.DataPreprocessor import DataPreprocessor
from data_preprocessing.processing.PipelineCreator import PipelineCreator
from data_preprocessing.processing.ManualPreprocessor import ManualPreprocessor

def manual_worker() -> None:
	print("Загрузка данных...")
	data = TitanicLoader()
	data.load_from_kaggle()
	print(data.get_info())

	print("\nПрепроцессинг данных...")
	preproc = DataPreprocessor(data.df)
	preproc.select_features(None)
	preproc.add_missing_values(None, None)
	X_train, X_test, y_train, y_test = preproc.split_data(None, None)
	print(preproc.get_split_info())

	print("\nСоздание и обучение ручного препроцессора...")
	manual_processor = ManualPreprocessor(
		numeric_features=None,
		categorical_features=None,
		classifier=None
	)

	accuracy = manual_processor.process_and_train(
		X_train,
		y_train,
		X_test,
		y_test)

	print("\nОценка качества модели (ручной подход)...")
	print(f" Точность модели: {accuracy:.6f}")
	print(f" Процент правильных предсказаний: {accuracy * 100:.4f}%")

def pipeline_worker() -> None:
    print("Загрузка данных...")
    data = TitanicLoader()
    data.load_from_kaggle()
    print(data.get_data())
    print(data.get_info())

    print("\nПрепроцессинг данных...")
    preproc = DataPreprocessor(data.df)
    preproc.select_features(None)
    preproc.add_missing_values(None, None)
    X_train, X_test, y_train, y_test = preproc.split_data(None, None)
    print(preproc.get_split_info())

    print("\nСоздание и обучение pipeline...")
    pipeline_creator = PipelineCreator(
        numeric_features=None,
        categorical_features=None,
        classifier=None
    )
    pipeline = pipeline_creator.create_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nОценка качества модели...")
    y_predicted = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(f" Точность модели: {accuracy:.6f}")
    print(f" Процент правильных предсказаний: {accuracy * 100:.4f}%")

def measure_time(func: Callable[..., Any], iteration_count: int = 20, *args, **kwargs) -> float:
    """ Считает время затраты на обучение ручным способом, и с помощью pipeline. """
    total_time = 0

    for i in range(0, iteration_count):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time += (time.time() - start_time)
    
    return total_time / iteration_count

def main():
    pipeline_worker()
    #manual_worker()
     
    #manual_time = measure_time(manual_worker, 20)
    #print(f"Время затраты на ручное обучение на 20 опытах: {manual_time:.6f}")
    
    #pipeline_time = measure_time(pipeline_worker, 20)
    #print(f"Время затраты на pipeline обучение на 20 опытах: {pipeline_time:.6f}")
          

if __name__ == '__main__':
    main()