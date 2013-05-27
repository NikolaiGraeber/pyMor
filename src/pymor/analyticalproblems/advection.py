# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pymor.core as core
from pymor.tools import Named
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction


class InstationaryAdvectionProblem(core.BasicInterface, Named):

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 flux_function=ConstantFunction(value=[0, 0], dim_domain=2, dim_range=2),
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2),
                 initial_data=ConstantFunction(dim_domain=2), T=1, name=None):
        self.domain = domain
        self.rhs = rhs
        self.flux_function = flux_function
        self.dirichlet_data = dirichlet_data
        self.initial_data = initial_data
        self.T = T
        self.name = name
