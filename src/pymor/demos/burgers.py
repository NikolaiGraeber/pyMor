#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Thermalblock demo.

Usage:
  burgers.py [-hp] [--grid=NI] [--nt=COUNT] [--plot-solutions] [--test=COUNT]
             EXP_MIN EXP_MAX SNAPSHOTS RBSIZE


Arguments:
  EXP_MIN    Minimal exponent
  EXP_MAX    Maximal exponent
  SNAPSHOTS  Number of snapshots for basis generation.

  RBSIZE     Size of the reduced basis


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --nt=COUNT             Number of time steps [default: 100].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

import numpy as np
from docopt import docopt

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
from pymor.analyticalproblems.burgers import BurgersProblem
from pymor.discretizers.advection import discretize_instationary_advection_fv
from pymor.reductors import reduce_generic_rb
from pymor.algorithms import greedy
from pymor.algorithms.basisextension import pod_basis_extension
core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')


def burgers_demo(args):
    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])
    args['EXP_MIN'] = int(args['EXP_MIN'])
    args['EXP_MAX'] = int(args['EXP_MAX'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])

    print('Solving on RectGrid(({0},{0}))'.format(args['--grid']))

    print('Setup Problem ...')
    problem = BurgersProblem(parameter_range=(args['EXP_MIN'], args['EXP_MAX']))

    print('Discretize ...')
    discretization, _ = discretize_instationary_advection_fv(problem, diameter=m.sqrt(2) / args['--grid'],
                                                             nt=args['--nt'])

    print(discretization.parameter_info())

    if args['--plot-solutions']:
        print('Showing some solutions')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            discretization.visualize(U)


    print('RB generation ...')

    reductor = reduce_generic_rb
    extension_algorithm = partial(pod_basis_extension)
    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         use_estimator=False, error_norm=lambda U: np.max(discretization.l2_norm(U)),
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']


    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    l2_err_max = -1
    cond_max = -1
    for mu in discretization.parameter_space.sample_randomly(args['--test']):
        print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
        URB = reconstructor.reconstruct(rb_discretization.solve(mu))
        U = discretization.solve(mu)
        l2_err = np.max(discretization.l2_norm(U - URB))
        if l2_err > l2_err_max:
            l2_err_max = l2_err
            Umax = U
            URBmax = URB
            mumax = mu
        print('L2-error = {}'.format(l2_err))
    toc = time.time()
    t_est = toc - tic
    real_rb_size = len(greedy_data['data'])

    print('''
    *** RESULTS ***

    Problem:
       parameter range:                    ({args[EXP_MIN]}, {args[EXP_MAX]})
       h:                                  sqrt(2)/{args[--grid]}
       nt:                                 {args[--nt]}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal L2-error:                   {l2_err_max}  (mu = {mumax})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-err']:
        discretization.visualize(U - URB)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    burgers_demo(args)
