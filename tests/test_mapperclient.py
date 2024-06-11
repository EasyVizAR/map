import numpy as np

from map import mapperclient


def test_cylinder_contains_any():
    points = []
    x = np.array([0, 0, 0])
    assert not mapperclient.cylinder_contains_any(points, x)

    points = np.array([[10, 10, 10]])
    assert not mapperclient.cylinder_contains_any(points, x)

    points = np.array([[10, 10, 10], x])
    assert mapperclient.cylinder_contains_any(points, x)
