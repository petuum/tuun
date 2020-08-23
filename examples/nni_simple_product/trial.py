import numpy as np
import nni

f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[1])

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin1 = x['suggestion'][0]['__ndarray__'][0]
xin2 = x['suggestion'][1]['__ndarray__'][0]
xin = [xin1, xin2]
print('xin = {}'.format(xin))

print('querying function now')
y = f(xin)
print('y = {}'.format(y))
print('-----')

nni.report_final_result(y)
