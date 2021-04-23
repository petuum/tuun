import numpy as np
import nni

f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = list(x.values())
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
