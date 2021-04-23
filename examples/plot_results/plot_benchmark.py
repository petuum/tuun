"""
Plot results on tuning two benchmark functions.
"""

import json
import pathlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import neatplot
neatplot.set_style()


benchmarks = {
    'hartmann': 'Hartmann 6 Dimensions',
    'branin40': 'Branin 40 Dimensions',
}

iterations = {
    'hartmann': 60,
    'branin40': 120,
}

methods = {
    'random': 'Random Search',
    'gp': 'GP',
    'tpe': 'TPE',
    'smac': 'SMAC',
    'tuun': 'Tuun',
}

trials = {
    'random': {'hartmann': [0, 1, 2, 3, 4], 'branin40': [0, 1, 2, 3, 4]},
    'gp': {'hartmann': [0, 1, 2, 3, 4], 'branin40': [0, 1, 2, 3, 4]},
    'tpe': {'hartmann': [0, 1, 2, 3, 4], 'branin40': [0, 1, 2, 3, 4]},
    'smac': {'hartmann': [0, 1, 2, 3, 4], 'branin40': [0, 1, 2, 3, 4]},
    'tuun': {'hartmann': [0, 1, 2, 3, 4], 'branin40': [0, 1, 2, 3, 4]},
}

savefig = True

lims = {'hartmann': [(-1, 61), (-3.5, 0.1)], 'branin40': [(-1, 121), (50, 1050)]}


for benchmark in benchmarks.keys():

    # reset color cycle
    clist = rcParams['axes.prop_cycle']
    cgen = itertools.cycle(clist)

    handles = []

    for method in methods.keys():

        y_list = []
        y_bsf_list = []

        for trial in trials[method][benchmark]:
            file_name = f'{benchmark}_{method}_{trial}.json'
            file_path = f'examples/plot_results/results_benchmark/' + file_name

            with open(file_path) as f:
                result = json.load(f)

            y_raw = [float( res['value']) for res in result]
            y = np.array(y_raw).reshape(-1)
            
            # equalize length
            if len(y) < iterations[benchmark]:
                y = np.append(y, np.repeat(y[-1], iterations[benchmark] - len(y)))

            y_bsf = np.minimum.accumulate(y)

            y_list.append(y)
            y_bsf_list.append(y_bsf)


        x = np.arange(iterations[benchmark])
        y_bsf = np.mean(y_bsf_list, 0)
        y_bsf_err = np.std(y_bsf_list, 0) / np.sqrt(len(y_bsf_list))


        linecolor = next(cgen)['color']

        plt.fill_between(
            x, y_bsf - y_bsf_err, y_bsf + y_bsf_err, color=linecolor, alpha=0.1
        )
        h, = plt.plot(
            x,
            y_bsf,
            linewidth=1.5,
            linestyle='-',
            label=methods[method],
            color=linecolor,
        )

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
