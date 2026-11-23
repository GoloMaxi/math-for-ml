import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mathml.vectors import cosine_distance, norm, pairwise_distances


ATOL = 1e-9  # Допустимая абсолютная ошибка по условию задания.


def test_pairwise_euclidean():
    # Две точки в a и три в b — ожидаем шесть расстояний.
    a = np.array([[0.5, -1.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [0.0, 0.0], [-0.5, 2.0]])

    actual = pairwise_distances(a, b)
    expected = cdist(a, b, metric="euclidean")

    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("order", [1, 2, np.inf])
def test_norm_matches_numpy(order):
    vector = np.array([-7.0, 0.5, 2.0])

    actual = norm(vector, order)
    expected = np.linalg.norm(vector, ord=order)

    assert actual == pytest.approx(expected, abs=ATOL, rel=0)


def test_norm_rejects_unsupported_order():
    vector = np.array([-3.0, 4.0])

    with pytest.raises(ValueError):
        norm(vector, 3)


def test_cosine_distance_matches_scipy():
    a = np.array([1.0, -2.0])
    b = np.array([3.0, 4.0])

    actual = cosine_distance(a, b)

    # cdist ожидает матрицы: превращаем каждый вектор в одну строку.
    # Результат имеет форму (1, 1), поэтому берём число из ячейки [0, 0].
    expected = cdist(a[None, :], b[None, :], metric="cosine")[0, 0]

    assert actual == pytest.approx(expected, abs=ATOL, rel=0)


@pytest.mark.parametrize("zero_first", [True, False])
def test_cosine_distance_rejects_zero_vector(zero_first):
    zero = np.array([0.0, 0.0])
    vector = np.array([1.0, -1.0])
    a, b = (zero, vector) if zero_first else (vector, zero)

    with pytest.raises(ValueError):
        cosine_distance(a, b)