def validate_null_object(obj, message="Объект является null"):
    """ Проверяет что объект obj является null.
        Если объект является null, выбрасывается Exception.
        Если объект является не null, то возвращается obj.
    """
    if obj is None:
        raise Exception(message, obj)
