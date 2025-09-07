from sklearn.metrics import accuracy_score
from data_preprocessing.data.TitanicLoader import TitanicLoader
from data_preprocessing.processing.DataPreprocessor import DataPreprocessor
from data_preprocessing.processing.PipelineCreator import PipelineCreator

def main():
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
    print(f" Точность модели: {accuracy:.4f}")
    print(f" Процент правильных предсказаний: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()