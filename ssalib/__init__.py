"""Visual and Automated Singular Spectrum Analysis Library (VASSAL)

Author: Damien Delforge <damien.delforge@adscian.be>
        Alice Alonso <alice.alonso@adscian.be>
License: BSD-3-Clause

"""
from .ssa import SingularSpectrumAnalysis
from .montecarlo_ssa import MonteCarloSSA

__version__ = '0.1.0a1'
__author__ = ('Damien Delforge <damien.delforge@adscian.be>, '
              'Alice Alonso <alice.alonso@adscian.be>')
__license__ = 'BSD-3-Clause'
__all__ = ['MonteCarloSSA', 'SingularSpectrumAnalysis']
