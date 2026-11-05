import pytest

from algo_base import Param


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2.5, True, "3"])
def test_integer_parameter_rejects_unsafe_api_inputs(value):
    parameter = Param("epochs", "Époques", "int", 3, minimum=1)
    with pytest.raises(ValueError):
        parameter.validate(value)


def test_parameter_defaults_respect_both_bounds():
    with pytest.raises(ValueError, match="minimum"):
        Param("epochs", "Époques", "int", 0, minimum=1)
    with pytest.raises(ValueError, match="maximum"):
        Param("epochs", "Époques", "int", 11, minimum=1, maximum=10)


def test_numeric_and_choice_parameters_validate_requests():
    assert Param("epochs", "Époques", "int", 2, minimum=1).validate(3.0) == 3
    choice = Param("loss", "Perte", "choice", "mse", choices=[("MSE", "mse")])
    with pytest.raises(ValueError, match="choix invalide"):
        choice.validate("unavailable")
