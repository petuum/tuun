import numpy as np
import nni

f = lambda x: x ** 4 - x ** 2 + 0.1 * x

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = x['suggestion']['__ndarray__'][0]
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
