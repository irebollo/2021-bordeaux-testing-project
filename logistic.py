import numpy as np

def f(x, r):
	result = r * x * (1 - x)
	return result
	
def iterate_f(it, x, r):
	result = []
	new_x = x
	for i in range(it):
		new_x = f(new_x, r)
		result.append(new_x)
	return np.array(result)
