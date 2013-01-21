from __future__ import absolute_import, division, print_function
from nose.tools import raises
import mock

from pymor.core.interfaces import (BasicInterface, contract, abstractmethod, abstractstaticmethod, 
                                   abstractclassmethod)
from pymor.core import exceptions
from pymor.core import timing
from pymor.core.exceptions import ContractNotRespected
from pymor.grids import AllDirichletBoundaryInfo as ADIA
from pymor.playground.boundaryinfos.oned import AllDirichletBoundaryInfo as ADIB
import pymor.grids.boundaryinfos
import pymor.playground.boundaryinfos.oned
from pymortests.base import TestBase

class UnknownInterface(BasicInterface):
    pass

class StupidInterface(BasicInterface):
    '''I am a stupid Interface'''

    @contract
    @abstractmethod
    def shout(self, phrase, repeat):
        ''' I repeatedly print a phrase.
        
        :param phrase: what I'm supposed to shout
        :param repeat: how often I'm shouting phrase
        :type phrase: str
        :type repeat: int
        
        .. seealso:: blabla
        .. warning:: blabla
        .. note:: blabla
        '''
        pass

class BrilliantInterface(BasicInterface):
    '''I am a brilliant Interface'''

    @contract
    @abstractmethod
    def whisper(self, phrase, repeat):
        '''
        :type phrase: str
        :type repeat: int,=1
        '''
        pass
    
class StupidImplementer(StupidInterface):

    def shout(self, phrase, repeat):
        print(phrase*repeat)

class AverageImplementer(StupidInterface, BrilliantInterface):

    def shout(self, phrase, repeat):
        #cannot change docstring here or else
        print(phrase*repeat)

    def whisper(self, phrase, repeat):
        print(phrase*repeat)

class DocImplementer(AverageImplementer):
    '''I got my own docstring'''

    @contract
    def whisper(self, phrase, repeat):
        '''my interface is stupid, I can whisper a lot more
        Since I'm overwriting an existing contract, I need to be decorated anew.

        :type phrase: str
        :type repeat: int,>0
        '''
        self.logger.critical(phrase*repeat)
        
class FailImplementer(StupidInterface):
    pass

class InterfaceTest(TestBase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFreeze(self):
        b = AverageImplementer()
        b.level = 43
        b.lock()
        b.level = 41
        with self.assertRaises(exceptions.ConstError):
            b.new = 42
        b.freeze()
        with self.assertRaises(exceptions.ConstError):
            b.level = 0
        b.freeze(False)
        b.level = 0
        b.lock(False)
        b.level = 0

    @raises(ContractNotRespected)
    def testContractFail(self):
        AverageImplementer().whisper('Wheee\n', -2)

    def testContractSuccess(self):
        AverageImplementer().shout('Wheee\n', 6)
        
    def testImplementorlist(self):
        imps = ['StupidImplementer', 'AverageImplementer', 'FailImplementer']
        self.assertEqual(imps, StupidInterface.implementor_names(), '')
        self.assertEqual(imps + ['DocImplementer'], StupidInterface.implementor_names(True), '')
        self.assertEqual(['AverageImplementer'], BrilliantInterface.implementor_names(), '')
        
    def testAbstractMethods(self):
        class ClassImplementer(BasicInterface):
    
            @abstractclassmethod
            def abstract_class_method(cls):
                pass
    
        class StaticImplementer(BasicInterface):
            
            @abstractstaticmethod
            def abstract_static_method():
                pass
        
        class CompleteImplementer(ClassImplementer, StaticImplementer):
            def abstract_class_method(cls):
                return cls.__name__
            def abstract_static_method():
                return 0
            
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = FailImplementer()
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = ClassImplementer()
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = StaticImplementer()
        inst = CompleteImplementer()
        self.assertEqual(inst.abstract_class_method(), 'CompleteImplementer')
        self.assertEqual(inst.abstract_static_method(), 0)        

            
class TimingTest(TestBase):
    
    def testTimingContext(self):
        with timing.Timer('busywait',self.logger.info) as timer:
            timing.busywait(1000)
            
    @timing.Timer('busywait_decorator', TestBase.logger.info)
    def wait(self):
        timing.busywait(1000)
            
    def testTimingDecorator(self):        
        self.wait()
        
    def testTiming(self):
        timer = timing.Timer('busywait',self.logger.info)
        timer.start()
        timing.busywait(1000)
        timer.stop()
        self.logger.info('plain timing took %s seconds', timer.dt)

class BoringTestInterface(BasicInterface):
    pass

class BoringTestClass(BasicInterface):

    @contract
    def validate_interface(self, cls, other):
        '''If you want to contract check on a type defined in the same module you CANNOT use the absolute path 
        notation. For classes defined elsewhere you MUST use it. Only builtins and classes with
        UberMeta as their metaclass can be checked w/o manually defining a new contract type.
        
        :type cls: pymortests.core.BoringTestInterface
        :type other: pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo
        '''
        pass

    @contract
    def dirichletTest(self, dirichletA, dirichletB):
        '''I'm used in testing whether contracts can distinguish 
        between equally named classes in different modules
        
        :type dirichletA: pymor.grids.interfaces.BoundaryInfoInterface
        :type dirichletB: pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo
        '''        
        return dirichletA != dirichletB


class ContractTest(TestBase):
    
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
        dirichletA = pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid)
        dirichletB = pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo()
        _combo(dirichletA, dirichletB)
        dirichletA = ADIA(grid)
        dirichletB = ADIB()
        _combo(dirichletA, dirichletB)
        
    def test_custom_contract_types(self):
        inst = BoringTestClass()
        with self.assertRaises(exceptions.ContractNotRespected):
            grid = mock.Mock()
            inst.validate_interface(object(), pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid))
        inst.validate_interface(BoringTestInterface(), pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo())
        
if __name__ == "__main__":
    import nose
    nose.core.runmodule(name='pymortests.core')