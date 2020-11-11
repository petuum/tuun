import numpy as np
import nni


# Proxy function 
str_map = {'a': [4], 'b': [5], 'c': [0], 'd': [0.7], 'e': [-0.7]}
f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
f = lambda x_list: f_s(str_map[x_list[0]]) + f_s(x_list[1])

x = nni.get_next_parameter()
print('x = {}'.format(x))

xin = list(x.values())
print('xin = {}'.format(xin))

print('Querying function now')
y = f(xin)
print('y = {}'.format(y))

nni.report_final_result(y)
