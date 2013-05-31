#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# make sure we got distribute in place
from distribute_setup import use_setuptools
use_setuptools()

import sys
import os
# not using this directly, but needs to be imported for nose not to fail
import multiprocessing
from subprocess import CalledProcessError, check_call
from setuptools import find_packages
from distutils.extension import Extension
from setuptools.dist import Distribution
from pkg_resources import working_set

_orig_generate_a_pyrex_source = None

tests_require = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive', 'tissue']
install_requires = ['distribute', 'numpy', 'scipy', 'PyContracts',
                    'docopt', 'dogpile.cache' , 'numpydoc']
setup_requires = ['cython', 'numpy', 'nose']
install_suggests = ['matplotlib', 'sympy'] + tests_require

class DependencyMissing(Exception):

    def __init__(self, names):
        super(DependencyMissing, self).__init__('Try: "for i in {} ; do pip install $i ; done"'.format(' '.join(names)))

class PipDistribution(Distribution):
    '''Overwrites the method responsible for installing missing eggs, substituting 
    a library call to easy_install with a syscall to pip
    '''
    
    def fetch_build_egg(self, req):
        # ideally this would include any version specs for given rec too
        cmd = ['pip', 'install', req.project_name]
        try:
            check_call(cmd)
        except CalledProcessError:
            return Distribution.fetch_build_egg(self, req)
        working_set.require(req.project_name)
        return working_set.find(req)
        
def _numpy_monkey():
    '''Apparently we need to monkey numpy's distutils to be able to build
    .pyx with Cython instead of Pyrex. The monkeying below is copied from
    https://github.com/matthew-brett/du-cy-numpy/blob/master/matthew_monkey.py
    via the discussion at http://comments.gmane.org/gmane.comp.python.numeric.general/37752
    '''
    global _build_src
    from os.path import join as pjoin, dirname
    from distutils.dep_util import newer_group
    from distutils.errors import DistutilsError

    from numpy.distutils.misc_util import appendpath
    from numpy.distutils import log

    from numpy.distutils.command import build_src
    _orig_generate_a_pyrex_source = build_src.build_src.generate_a_pyrex_source

    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        ''' Monkey patch for numpy build_src.build_src method

        Uses Cython instead of Pyrex, iff source contains 'pymor'

        Assumes Cython is present
        '''
        if not 'pymor' in source:
            return _orig_generate_a_pyrex_source(self, base, ext_name, source, extension)

        if self.inplace:
            target_dir = dirname(base)
        else:
            target_dir = appendpath(self.build_src, dirname(base))
        target_file = pjoin(target_dir, ext_name + '.c')
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target_file, 'newer'):
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=extension.include_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source,
                                                    options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling %r with Cython" \
                    % (cython_result.num_errors, source))
        return target_file

    build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source

def write_version():
    filename = os.path.join(os.path.dirname(__file__), 'src', 'pymor', 'version.py')
    try:
        revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*']).strip()
        with open(filename, 'w') as out:
            out.write('revstring = \'{}\''.format(revstring))
    except:
        if os.path.exists(filename):
            loc = {}
            execfile(filename, loc, loc)
            revstring = loc['revstring']
        else:
            revstring = '0.0.0-0-0'
    return revstring

def _setup(**kwargs):
    '''we'll make use of Distribution's __init__ downloading setup_requires packages right away here'''
    dist = PipDistribution(kwargs)
    
    # apparently requires is either int(0) or a callable that returns list(Requirement)
    reqs = dist.requires
    if not isinstance(reqs, int):
        for req in dist.requires():
            working_set.resolve(req, installer=dist.fetch_build_egg)

    # now that we supposedly have at least numpy + cython installed, use them
    # they're dropped in cwd as egg-dirs however. let's discover those first
    from pkg_resources import require
    require("Cython")
    require("numpy")
    _numpy_monkey()
    import Cython.Distutils
    cmdclass = {}#{'build_ext': Cython.Distutils.build_ext}
    from numpy import get_include
    ext_modules = [Extension("pymor.tools.relations", ["src/pymor/tools/relations.pyx"], include_dirs=[get_include()]),
                   Extension("pymor.tools.inplace", ["src/pymor/tools/inplace.pyx"], include_dirs=[get_include()])]
    kwargs['cmdclass'] = cmdclass
    kwargs['ext_modules'] = ext_modules

    # lastly we'll need to tweak matplotlibs config dir or else
    # installing it from setup will result in a SandboxViolation
    import os
    os.environ['MPLCONFIGDIR'] = "."

    from numpy.distutils.core import setup
    return setup(**kwargs)

def _missing(names):
    for name in names:
        try:
            __import__(name)
        except ImportError:
            yield name

def check_pre_require():
    '''these are packages that need to be present before we start out setup, because
    distribute/distutil/numpy.distutils makes automatic installation too unreliable
    '''
    missing = list(_missing(['numpy', 'scipy']))
    if len(missing):
        raise DependencyMissing(missing)


def setup_package():
#     check_pre_require()
    revstring = write_version()

    _setup(
        name='pyMor',
        version=revstring,
        author='pyMor developers',
        author_email='pymor-dev@listserv.uni-muenster.de',
        maintainer='Rene Milk',
        maintainer_email='rene.milk@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        scripts=['bin/%s' % n for n in [] ] + ['run_tests.py', 'distribute_setup.py'],
        url='http://pymor.org',
        description=' ' ,
        long_description=open('README.txt').read(),
        setup_requires=setup_requires,
        tests_require=tests_require,
        install_requires=install_requires,
        classifiers=['Development Status :: 3 - Alpha',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Visualization'],
        license='LICENSE.txt',
        zip_safe=False,
        test_suite='pymortests.base.suite',
    )

    missing = list(_missing(install_suggests))
    if len(missing):
        print('\n{0}\nThere are some suggested packages missing, try\nfor i in {1} ; do pip install $i ; done\n{0}'
              .format('*' * 79, ' '.join(missing)))


if __name__ == '__main__':
    setup_package()

