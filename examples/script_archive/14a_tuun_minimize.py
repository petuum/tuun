from tuun.main import Tuun

config = {
    'seed': 12,
    'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    #'normalize_real': True,
    #'model_config': {'name': 'standistmatgp'},
}
tu = Tuun(config)

# Update Tuun using set_config_from_search_space_list
#search_space_list = [('real', [[-20, 20]])]
#search_space_list = [('real', [[-20, 20]]), ('list', ['a', 'b', 'c', 'd', 'e'])]
#search_space_list = [('real', [[-20, 20]]), ('list', ['a', 'b', 'c', 'd', 'e']), ('list', ['a', 'b', 'c', 'd', 'e'])]
#search_space_list = [('real', [[-40, 40]]), ('list', ['a', 'b', 'c', 'd', 'e'])]
# ---
#search_space_list = [('real', [-20, 20]), ('list', ['a', 'b', 'c', 'd', 'e'])]
#search_space_list = [('real', [-20, 20]), ('list', ['a', 'b', 'c', 'd', 'e']), ('real', [-20, 20])]
search_space_list = [('list', ['a', 'b', 'c', 'd', 'e']), ('real', [-2, 2]), ('real', [-20, 20])]
#search_space_list = [('list', ['a', 'b', 'c', 'd', 'e']), ('real', [-2, 2]), ('list', ['cat', 'dog']), ('real', [-20, 20])]
#search_space_list = [('real', [-20, 20])]
#search_space_list = ('real', [-20, 20])
#search_space_list = ('list', ['a', 'b', 'c', 'd', 'e'])

tu.set_config_from_list(search_space_list)

def f(x_list):
    str_map = {'a': [4], 'b': [5], 'c': [0], 'd': [0.7], 'e': [-0.7]}
    f_s = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
    if len(x_list) == 1:
        return f_s(x_list)
    else:
        return f_s(x_list[0]) + f_s(str_map[x_list[1]])

def f(x_list):
    str_map = {'a': 4, 'b': 5, 'c': 0, 'd': 0.7, 'e': -0.7}
    f_s = lambda x: x ** 4 - x ** 2 + 0.1 * x
    if len(x_list) == 1:
        return f_s(x_list[0])
        #return f_s(str_map[x_list[0]])
    else:
        #return f_s(x_list[0]) + f_s(str_map[x_list[1]])
        return f_s(x_list[1]) + f_s(str_map[x_list[0]])

result = tu.minimize_function(f, 50)
