"""
Neat plotting utilities for Python.
"""

import pathlib
import matplotlib.pyplot as plt

def set_style(style_str='default'):
    if style_str == 'default':
        plt.style.use((pathlib.Path(__file__).parent / 'matplotlibrc').resolve())
    elif style_str == 'fonts':
        plt.style.use((pathlib.Path(__file__).parent / 'matplotlibrc_fonts').resolve())

def save_figure(file_name='figure', ext_list = ['pdf', 'png']):
    """Save figure for all extensions in ext_list."""

    if isinstance(ext_list, str):
        ext_list = [ext_list]

    for ext in ext_list:
        save_str = file_name + '.' + ext
        plt.savefig(save_str, bbox_inches='tight')
        print(f'Saved figure {save_str}')
