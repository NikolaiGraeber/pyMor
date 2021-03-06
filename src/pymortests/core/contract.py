# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import contracts
import mock
from nose.tools import raises

from pymortests.base import TestBase, runmodule
from pymortests.core.dummies import (AllDirichletBoundaryInfo, AverageImplementer, BoringTestClass)
from pymor.core.exceptions import ContractNotRespected
import pymor.grids.boundaryinfos
from pymor.core import exceptions
from pymor.core.interfaces import (contract,)

from pymor.grids import AllDirichletBoundaryInfo as ADIA


class ContractTest(TestBase):

    @raises(ContractNotRespected)
    def testContractFail(self):
        AverageImplementer().whisper('Wheee\n', -2)

    def testContractSuccess(self):
        AverageImplementer().shout('Wheee\n', 6)

    def testNaming(self):
        imp = BoringTestClass()

        def _combo(dirichletA, dirichletB):
            self.assertTrue(imp.dirichletTest(dirichletA, dirichletB))
            with self.assertRaises(ContractNotRespected):
                imp.dirichletTest(dirichletA, dirichletA)
            with self.assertRaises(ContractNotRespected):
                imp.dirichletTest(dirichletB, dirichletA)
            with self.assertRaises(ContractNotRespected):
                imp.dirichletTest(dirichletA, 1)
        grid = mock.Mock()
        dirichletB = AllDirichletBoundaryInfo()
        dirichletA = pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid)
        _combo(dirichletA, dirichletB)
        dirichletA = ADIA(grid)
        _combo(dirichletA, dirichletB)

    def test_custom_contract_types(self):
        inst = BoringTestClass()
        with self.assertRaises(exceptions.ContractNotRespected):
            grid = mock.Mock()
            inst.validate_interface(object(), pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid))

    def test_disabled_contracts(self):
        contracts.disable_all()

        @contract
        def disabled(phrase):
            '''
            :type phrase: str
            '''
            return phrase
        # this should not throw w/ contracts disabled
        disabled(int(8))
        contracts.enable_all()
        # this will still not throw because the disabled value is checked at decoration time only
        disabled(int(8))

        @contract
        def enabled(phrase):
            '''
            :type phrase: str
            '''
            return phrase
        # a newly decorated function will throw
        with self.assertRaises(exceptions.ContractNotRespected):
            enabled(int(8))

if __name__ == "__main__":
    runmodule(name='pymortests.core.contract')
