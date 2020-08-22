"""
A single query of branin function.
"""

import numpy as np
import nni

from examples.nni_branin.branin import branin


f = branin

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = x['suggestion']['__ndarray__']
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
