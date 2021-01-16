from tuun.main import Tuun

tu = Tuun()

#search_space = ('real', [[-5, 5]])
#search_space = [('real', [[-5, 5]]), ]
#search_space = ('real', [[-5, 5], [-4, 10]])
search_space = [('real', [[-5, 5], [-4, 10]]), ('list', ['red', 'green', 'blue'])]

tu.set_config_from_list(search_space)

#f = lambda x: x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]
def f(x):
    print(f'x = {x}')
    return x[0] ** 4 - x[0] ** 2 + 0.1 * x[0]

result = tu.minimize_function(f, 20)
