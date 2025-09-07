from data_preprocessing.helpers.validationHelper import validate_null_object

def set_default_if_input_value_null(default_value, input_value):
    """Возвращает default_value, если input_value является None."""

    validate_null_object(default_value)

    if input_value is None:
        return default_value

    return input_value