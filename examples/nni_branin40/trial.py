"""
A single query of branin function.
"""

import numpy as np
import nni

from examples.nni_branin40.branin import branin


f = lambda x: np.sum([branin(x[2 * i : 2 * i + 2]) for i in range(20)])

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = x['suggestion']
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
