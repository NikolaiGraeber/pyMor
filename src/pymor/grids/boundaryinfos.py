# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import BoundaryInfoInterface
from pymor.domaindescriptions import BoundaryType


class EmptyBoundaryInfo(BoundaryInfoInterface):
    '''`BoundaryInfo` without any `BoundaryTypes`.
    '''

    def __init__(self, grid):
        super(EmptyBoundaryInfo, self).__init__()
        self.grid = grid
        self.boundary_types = set()

    def mask(self, boundary_type, codim):
        assert False, ValueError('Has no boundary_type "{}"'.format(boundary_type))


class BoundaryInfoFromIndicators(BoundaryInfoInterface):
    '''`BoundaryInfo` where the `BoundaryTypes` are determined by indicator functions.

    Parameters
    ----------
    grid
        The grid to which the `BoundaryInfo` is associated.
    indicators
        dict where each key is a `BoundaryType` and the corresponding value is a boolean
        valued function on the analytical domain indicating if a point belongs to a boundary
        of the `BoundaryType`. (The indicator functions must be vectorized.)
    '''

    def __init__(self, grid, indicators):
        super(BoundaryInfoFromIndicators, self).__init__()
        self.grid = grid
        self.boundary_types = indicators.keys()
        self._masks = {boundary_type: [np.zeros(grid.size(codim), dtype='bool') for codim in xrange(1, grid.dim + 1)]
                       for boundary_type in self.boundary_types}
        for boundary_type, codims in self._masks.iteritems():
            for c, mask in enumerate(codims):
                mask[grid.boundaries(c + 1)] = indicators[boundary_type](grid.centers(c + 1)[grid.boundaries(c + 1)])

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        return self._masks[boundary_type][codim - 1]


class AllDirichletBoundaryInfo(BoundaryInfoInterface):
    '''`BoundaryInfo` where each boundray entity has `BoundaryType('dirichlet')`.
    '''

    def __init__(self, grid):
        super(AllDirichletBoundaryInfo, self).__init__()
        self.grid = grid
        self.boundary_types = set((BoundaryType('dirichlet'),))

    def mask(self, boundary_type, codim):
        assert boundary_type == BoundaryType('dirichlet'), ValueError('Has no boundary_type "{}"'.format(boundary_type))
        assert 1 <= codim <= self.grid.dim
        return np.ones(self.grid.size(codim), dtype='bool') * self.grid.boundary_mask(codim)


class SubGridBoundaryInfo(BoundaryInfoInterface):

    def __init__(self, subgrid, grid, grid_boundary_info, new_boundary_type=None):
        assert new_boundary_type is None or isinstance(new_boundaries_type, BoundaryType)
        boundary_types = grid_boundary_info.boundary_types

        has_new_boundaries = False
        masks = []
        for codim in xrange(1, subgrid.dim + 1):
            parent_indices = subgrid.parent_indices(codim)[subgrid.boundaries(codim)]
            new_boundaries = np.where(grid.boundary_mask(codim)[parent_indices])
            if len(new_boundaries) > 0:
                has_new_boundaries = True
            m = {}
            for t in boundary_types:
                m[t] = grid_boundary_info.mask(t, codim)[subgrid.parent_indices(codim)]
                if t == new_boundary_type:
                    m[t][new_boundaries] = True
            if new_boundary_type is not None and new_boundary_type not in boundary_types:
                m[new_boundary_type] = np.zeros(subgrid.size(codim), dtype=np.bool)
                m[new_boundary_type][new_boundaries] = True
            masks.append(m)
        self.__masks = masks

        self.boundary_types = set(grid_boundary_info.boundary_types)
        if has_new_boundaries and new_boundary_type is not None:
            self.boundary_types.add(new_boundary_type)

    def mask(self, boundary_type, codim):
        assert 1 <= codim < len(self.__masks) + 1, 'Invalid codimension'
        assert boundary_type in self.boundary_types
        return self.__masks[codim - 1][boundary_type]
