"""Singular Spectrum Analysis Library (SSALib)
"""
from .ssa import SingularSpectrumAnalysis
from .montecarlo_ssa import MonteCarloSSA

__version__ = '0.1.2b1'
__author__ = ('Damien Delforge <damien.delforge@adscian.be>, '
              'Alice Alonso <alice.alonso@adscian.be>, '
              'Olivier de Viron, '
              'Marnik Vanclooster, '
              'Niko Speybroeck')
__license__ = 'BSD-3-Clause'
__all__ = ['MonteCarloSSA', 'SingularSpectrumAnalysis']
