import numpy as np
import pytest
from math import isclose
from logistic import f, iterate_f

@pytest.mark.parametrize("x,r,expected",[(0.1,2.2,0.198),
					  (0.2,3.4,0.544),
					  (0.75,1.7,0.31875)])
def test_logistic(x, r,expected):
	# When
	result = f(x, r)
	# Then
	assert isclose(result, expected)

@pytest.mark.parametrize("x, r, it, expected",[(0.1, 2.2, 1, [0.198]),
						 (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]),
						 (0.75, 1.7, 2, [0.31875, 0.369152])])
def test_logistic_iterated(x, r, it, expected):
	# When
	result = iterate_f(it, x, r)
	# Then
	np.testing.assert_allclose(result, np.array(expected), rtol=1e-5)

