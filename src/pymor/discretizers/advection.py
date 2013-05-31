# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.domaindiscretizers import discretize_domain_default
from pymor.operators.fv import NonlinearAdvectionLaxFriedrichs, L2Product
from pymor.operators import NumpyLinearOperator
from pymor.grids import RectGrid
from pymor.gui.qt import visualize_glumpy_patch
from pymor.discretizations import InstationaryNonlinearDiscretization
from pymor.la import induced_norm
from pymor.la import NumpyVectorArray


def discretize_instationary_advection_fv(analytical_problem, diameter=None, nt=100, domain_discretizer=None,
                                         grid=None, boundary_info=None):

    assert isinstance(analytical_problem, InstationaryAdvectionProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    L = NonlinearAdvectionLaxFriedrichs(grid, p.flux_function, lmbda=1)
    F = NumpyLinearOperator(np.zeros((1, L.dim_source)))
    I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
    I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
    I = NumpyVectorArray(I)

    def visualize(U):
        visualize_glumpy_patch(grid, U, bounding_box=grid.domain, codim=0)
        # assert len(U) == 1
        # grid.visualize(U.data.ravel())
        # import matplotlib.pyplot as pl
        # pl.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], grid.subentities(0, 2), U.data.ravel())
        # pl.colorbar()
        # pl.show()

    discretization = InstationaryNonlinearDiscretization(L, F, I, p.T, nt, visualizer=visualize,
                                                         name='{}_FV'.format(p.name))

    discretization.l2_product = L2Product(grid)
    discretization.l2_norm = induced_norm(discretization.l2_product)

    if hasattr(p, 'parameter_space'):
        discretization.parameter_space = p.parameter_space

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
