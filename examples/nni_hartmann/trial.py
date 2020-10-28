"""
A single query of branin function.
"""

import numpy as np
import nni

from examples.nni_hartmann.hartmann import hartmann6


f = hartmann6

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = x['suggestion']
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
