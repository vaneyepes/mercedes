import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InvalidParameterError

def test_valid_max_features_int():
    model = RandomForestRegressor(max_features=5)
    assert model.max_features == 5

def test_valid_max_features_float():
    model = RandomForestRegressor(max_features=0.5)
    assert model.max_features == 0.5

def test_valid_max_features_str_sqrt():
    model = RandomForestRegressor(max_features='sqrt')
    assert model.max_features == 'sqrt'

def test_valid_max_features_str_log2():
    model = RandomForestRegressor(max_features='log2')
    assert model.max_features == 'log2'

def test_invalid_max_features():
    with pytest.raises(InvalidParameterError):
        model = RandomForestRegressor(max_features='auto')

def test_error_score_raise():
    with pytest.raises(InvalidParameterError):
        model = RandomForestRegressor(max_features='auto', error_score='raise')