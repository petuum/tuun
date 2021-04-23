"""
Plot results on tuning two benchmark functions.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import neatplot
neatplot.set_style()


benchmarks = {
    'hartmann': 'Hartmann 6 Dimensions',
    'branin': 'Branin 40 Dimensions',
}

methods = {
    'random': 'Random Search',
    'gp': 'GP',
    'tpe': 'TPE',
    'smac': 'SMAC',
    'tuun': 'Tuun'
}

savefig = True

lims = {'hartmann': [(-1, 61), (-3.5, 0.1)], 'branin': [(-1, 121), (50, 1050)]}
#lims = {'hartmann': [(-1, 61), (-3.5, 0.1)], 'branin': [(10, 121), (50, 800)]}


for benchmark in benchmarks.keys():

    handles = []

    for method in methods.keys():
        file_name = f'{benchmark}_{method}.json'
        file_path = f'examples/plot_results/results_benchmark_old/' + file_name

        with open(file_path) as f:
            result = json.load(f)

        y_list = [float( res['value']) for res in result]
        y = np.array(y_list).reshape(-1)
        y_bsf = np.minimum.accumulate(y)

        x = np.arange(len(y))

        h, = plt.plot(x, y_bsf, linewidth=1.5, linestyle='-', label=methods[method])
        handles.append(h)

    plt.legend(handles=handles)

    plt.xlabel('Iteration')
    plt.ylabel('Objective $y$')

    plt.xlim(lims[benchmark][0])
    plt.ylim(lims[benchmark][1])

    plt.title(benchmarks[benchmark])

    if savefig:
        neatplot.save_figure(f'01_{benchmark}', ['pdf', 'png'])

    plt.show()
