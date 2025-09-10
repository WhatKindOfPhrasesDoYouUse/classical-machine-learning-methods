from typing import Any

def validate_null_object(obj, message="Объект является null"):
    """ Проверяет что объект obj является null.
        Если объект является null, выбрасывается Exception.
    """
    if obj is None:
        raise Exception(message, obj)

def validate_null_array_element(objects: list[Any]):
	""" Проверяет что в массиве объектов objects есть null значение.
		Если такое значение есть, выбрасывается ValueError.
	"""
	for index, obj in enumerate(objects):
		if obj is None:
			raise ValueError(f"Наден Null элемент по индексу: {index}")