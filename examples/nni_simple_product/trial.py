import numpy as np
import nni

f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = x['suggestion']
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
