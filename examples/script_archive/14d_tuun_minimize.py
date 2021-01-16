from tuun.main import Tuun

config = {
    'seed': 12,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb'},
    'model_config': {'name': 'standistmatgp'}
}
tu = Tuun(config)

# Update Tuun using set_config_from_search_space_list
#search_space_list = [('list', ['a', 'b', 'c', 'd', 'e']), ('real', [[-10, 10], [-4, 4]])]
search_space_list = [('real', [-10, 10]), ('list', ['a', 'b', 'c', 'd', 'e']), ('real', [-4, 4])]

tu.set_config_from_list(search_space_list)

#f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
#f = lambda x_list: f_s(x_list[1][0]) + f_s(x_list[1][1])

f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
f = lambda x_list: f_s(x_list[0]) + f_s(x_list[2])

result = tu.minimize_function(f, 50)
