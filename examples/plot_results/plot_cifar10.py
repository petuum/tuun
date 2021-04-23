"""
Plot results for PyTorch HPO tuning on CIFAR10.
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
    'cifar10': 'PyTorch HPO on CIFAR10',
}

iterations = {
    'cifar10': 60,
}

methods = {
    'random': 'Random Search',
    'tuun': 'Tuun',
}

trials = {
    'random': {'cifar10': [0, 1, 2, 3, 4]},
    'tuun': {'cifar10': [0, 1, 2, 3, 4]},
}

show_trials = False
savefig = True

lims = {'cifar10': [(0, 60), (0.8997, 0.95)]}


for benchmark in benchmarks.keys():

    # reset color cycle
    clist = rcParams['axes.prop_cycle']
    cgen = itertools.cycle(clist)

    handles = []

    for method in methods.keys():

        y_list = []
        y_bsf_list = []

        for trial in trials[method][benchmark]:
            file_name = f'{benchmark}_{method}{trial}.json'
            file_path = f'examples/plot_results/results_cifar10/' + file_name

            with open(file_path) as f:
                result = json.load(f)

            y_raw = [float( res['value']) for res in result]
            y = np.array(y_raw).reshape(-1)
            
            # equalize length
            if len(y) < iterations[benchmark]:
                y = np.append(y, np.repeat(y[-1], iterations[benchmark] - len(y)))

            y_bsf = np.maximum.accumulate(y)

            y_list.append(y)
            y_bsf_list.append(y_bsf)


        x = np.arange(iterations[benchmark])
        y_bsf = np.mean(y_bsf_list, 0)
        y_bsf_err = np.std(y_bsf_list, 0) / np.sqrt(len(y_bsf_list))


        linecolor = next(cgen)['color']

        # Plot error bands
        plt.fill_between(
            x, y_bsf - y_bsf_err, y_bsf + y_bsf_err, color=linecolor, alpha=0.1
        )
 
        # Plot trial curves
        if show_trials:
            for y_bsf_i in y_bsf_list:
                plt.plot(
                    x,
                    y_bsf_i,
                    linewidth=0.2,
                    linestyle='-',
                    label=methods[method],
                    color=linecolor,
                    alpha=0.4,
                )

        # Plot mean curves
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
