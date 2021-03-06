# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import sys
import nose


# TODO remove copypaste from pymortests.base
class PymorTestSelector(nose.selector.Selector):

    def __init__(self, *args, **kwargs):
        super(PymorTestSelector, self).__init__(*args, **kwargs)
        self._skip_grid = 'PYMOR_NO_GRIDTESTS' in os.environ

    def wantDirectory(self, dirname):
        return 'src' in dirname

    def wantFile(self, filename):
        if self._skip_grid and 'grid' in filename:
            return False
        return filename.endswith('.py') and ('pymortests' in filename or
                                             'dynamic' in filename)

    def wantModule(self, module):
        parts = module.__name__.split('.')
        return 'pymortests' in parts or 'pymor' in parts

    def wantClass(self, cls):
        ret = super(PymorTestSelector, self).wantClass(cls)
        if hasattr(cls, 'has_interface_name'):
            return ret and not cls.has_interface_name()
        return ret

if __name__ == '__main__':

    # cli = [__file__, '-vv', '--nologcapture', '-s', '--collect-only']  # useful to run nose plugins like tissue only
    cli = [__file__, ]
    cli.extend(sys.argv[1:])

    config_files = nose.config.all_config_files()
    config_files.append('setup.cfg')
    manager = nose.plugins.manager.DefaultPluginManager()
    cfg = nose.config.Config(files=config_files, plugins=manager)
    cfg.exclude = []
    selector = PymorTestSelector(config=cfg)
    loader = nose.loader.defaultTestLoader(config=cfg, selector=selector, workingDir='.')
    success = nose.core.run(argv=cli, testLoader=loader, config=cfg, module='pymortests')
    sys.exit(not success)
