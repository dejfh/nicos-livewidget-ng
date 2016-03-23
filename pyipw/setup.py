#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import platform
from os import path
from distutils.core import setup, Extension

from sipdistutils import build_ext as sip_build_ext
# from gitversion import get_git_version

try:
    from PyQt4 import pyqtconfig
except ImportError:
    from sipconfig import Configuration

    def query_var(varname):
        p = os.popen('qmake -query ' + varname, 'r')
        return p.read().strip()
    # new-style configured PyQt4, no pyqtconfig module
    from PyQt4.QtCore import PYQT_CONFIGURATION
    pyqt_sip_flags = PYQT_CONFIGURATION['sip_flags'].split()
    pyqt_sip_dir = path.join(Configuration().default_sip_dir, 'Py2-Qt4')
    moc_bin = path.join(query_var('QT_INSTALL_BINS'), 'moc')
    qt_inc_dir = query_var('QT_INSTALL_HEADERS')
    qt_lib_dir = query_var('QT_INSTALL_LIBS')
else:
    config = pyqtconfig.Configuration()
    pyqt_sip_flags = config.pyqt_sip_flags.split()
    moc_bin = config.build_macros()['MOC']
    pyqt_sip_dir = config.pyqt_sip_dir
    qt_inc_dir = config.qt_inc_dir
    qt_lib_dir = config.qt_lib_dir

class moc_build_ext(sip_build_ext):
    '''Build with moc-generated files '''

    def finalize_options(self):
        sip_build_ext.finalize_options(self)
        if isinstance(self.sip_opts, str):
            self.sip_opts = self.sip_opts.split(' ')

        self.sip_opts = self.sip_opts + pyqt_sip_flags # + pyqwt_sip_flags
        # get_git_version()  # create version.h and RELEASE-VERSION files

    def _sip_sipfiles_dir(self):
        return pyqt_sip_dir

    def swig_sources(self, sources, extension=None):
        # Create .moc files from headers
        ret = sip_build_ext.swig_sources(self, sources, extension)
        for source in sources:
            if not source.endswith('.cpp'):
                continue
            header = source[:-4] + '.h'
            if not path.isfile(header):
                continue
            dest = path.join(self.build_temp, 'moc_' + source)
            self.spawn([moc_bin] + ['-o', dest, header])
            if path.getsize(dest) == 0:
                continue
            ret.append(dest)
        return ret

extra_include_dirs = [".."]
extra_lib_dirs = [qt_lib_dir, '../out']
extra_libs = ["pyimageplotwidget", "qwt"]

extra_include_dirs.extend(path.join(qt_inc_dir, subdir)
        for subdir in ['', 'QtCore', 'QtGui'])

if sys.platform == 'darwin':
    extra_libs.extend(['QtCore', 'QtGui'])
    extra_include_dirs.append("/usr/local/cfitsio/include")
elif sys.platform == 'win32':
    extra_libs.extend(['QtCore4', 'QtGui4'])
else:
    extra_libs.extend(['QtCore', 'QtGui'])
    dist = platform.linux_distribution()[0].strip()  # old openSUSE appended a space here :(
    if dist == 'openSUSE':
        extra_include_dirs.extend(["/usr/include/libcfitsio0", "/usr/include/cfitsio"])
    elif dist == 'Fedora':
        extra_include_dirs.append("/usr/include/cfitsio")
    elif dist in ['Ubuntu', 'LinuxMint', 'debian', 'CentOS']:
        pass
    else:
        print("WARNING: Don't know where to find headers and libraries for your distribution")
        # still try to build with usable defaults

sources = []

setup(
    name='nicosimageplot',
    version='1',
    ext_modules=[
    Extension('nicosimageplot',
          ['pyipw.sip'] + sources,
          include_dirs=['.'] + extra_include_dirs,
          library_dirs=extra_lib_dirs,
          libraries=extra_libs,
          extra_compile_args=['-std=c++11', '-fopenmp', '-Wall', '-Wextra', '-pedantic'],
          extra_link_args=['-fopenmp'],
          ),
    ],
    cmdclass={'build_ext': moc_build_ext}
)
