#!/usr/bin/env python
import sys
import os
import shutil
import inspect
import pkgutil
import importlib
from subprocess import check_call, CalledProcessError
from configparser import ConfigParser
from setuptools import setup, find_packages, Extension, _install_setup_requires
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand

try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'tweakwcs')
DESCRIPTION = metadata.get('description', 'A package for correcting alignment '
                           'errors in WCS objects')
LONG_DESCRIPTION = metadata.get('long_description', 'README.rst')
LONG_DESCRIPTION_CONTENT_TYPE = metadata.get('long_description_content_type',
                                             'text/x-rst')
AUTHOR = metadata.get('author', 'Mihai Cara')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://github.com/spacetelescope/tweakwcs')
LICENSE = metadata.get('license', 'BSD-3-Clause')

# load long description
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, LONG_DESCRIPTION), encoding='utf-8') as f:
    long_description = f.read()

if not pkgutil.find_loader('relic'):
    relic_local = os.path.exists('relic')
    relic_submodule = (relic_local and
                       os.path.exists('.gitmodules') and
                       not os.listdir('relic'))
    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/spacetelescope/relic.git'])

        sys.path.insert(1, 'relic')
    except CalledProcessError as e:
        print(e)
        exit(1)

import relic.release

version = relic.release.get_info()
if not version.date:
    default_version = metadata.get('version', '')
    default_version_date = metadata.get('version-date', '')
    version = relic.git.GitVersion(
        pep386=default_version,
        short=default_version,
        long=default_version,
        date=default_version_date,
        dirty=True,
        commit='',
        post='-1'
    )
relic.release.write_template(version,  os.path.join(*PACKAGENAME.split('.')))

# Install packages required for this setup to proceed:
SETUP_REQUIRES = [
    'numpy',
]

_install_setup_requires(dict(setup_requires=SETUP_REQUIRES))

for dep_pkg in SETUP_REQUIRES:
    try:
        importlib.import_module(dep_pkg)
    except ImportError:
        print("{0} is required in order to install '{1}'.\n"
              "Please install {0} first.".format(dep_pkg, PACKAGENAME),
              file=sys.stderr)
        exit(1)

def get_transforms_data():
    # Installs the schema files in jwst/transforms
    # Because the path to the schemas includes "stsci.edu" they
    # can't be installed using setuptools.
    transforms_schemas = []
    root = os.path.join(PACKAGENAME, 'schemas')

    if not os.path.isdir(root):
        return transforms_schemas

    for node, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith('.yaml'):
                transforms_schemas.append(
                    os.path.relpath(os.path.join(node, fname), root))
    # In the package directory, install to the subdirectory 'schemas'
    transforms_schemas = [os.path.join('schemas', s) for s in transforms_schemas]
    return transforms_schemas

PACKAGE_DATA = {
    '': [
        'README.rst',
        'LICENSE.txt',
        'CHANGELOG.rst',
        '*.fits',
        '*.txt',
        '*.inc',
        '*.cfg',
        '*.csv',
        '*.yaml',
        '*.json'
        ],
    'tweakwcs': get_transforms_data()
}

# Setup C module include directories
import numpy
include_dirs = [numpy.get_include()]

# Setup C module macros
define_macros = [('NUMPY', '1')]

# Handle MSVC `wcsset` redefinition
if sys.platform == 'win32':
    define_macros += [
        ('_CRT_SECURE_NO_WARNING', None),
        ('__STDC__', 1)
    ]


class InstallCommand(install):
    """Ensure tweakwcs's C extensions are available when imported relative
    to the documentation, instead of relying on `site-packages`. What comes
    from `site-packages` may not be the same tweakwcs that was *just*
    compiled.
    """
    def run(self):
        build_cmd = self.reinitialize_command('build_ext')
        build_cmd.inplace = 1
        self.run_command('build_ext')

        # Explicit request for old-style install?  Just do it
        if self.old_and_unmanageable or self.single_version_externally_managed:
            install.run(self)
        elif not self._called_from_setup(inspect.currentframe()):
            # Run in backward-compatibility mode to support bdist_* commands.
            install.run(self)
        else:
            self.do_egg_install()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['tweakwcs/tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


INSTALL_REQUIRES=[
    'numpy',
    'astropy>=3.1',
    'gwcs',
    'stsci.stimage',
    'stsci.imagestats',
    'spherical_geometry',
]

setup(
    name=PACKAGENAME,
    version=version.pep386,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Development Status :: 3 - Alpha',
    ],
    python_requires='>=3.5',
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=['pytest'],
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    ext_modules=[
        Extension('tweakwcs.chelp',
                  ['src/carrutils.c'],
                  include_dirs=[numpy.get_include()],
                  define_macros=define_macros),
    ],
    cmdclass={
        'test': PyTest,
        'install': InstallCommand,
        },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/tweakwcs/issues/',
        'Source': 'https://github.com/spacetelescope/tweakwcs/',
        'Help': 'https://hsthelp.stsci.edu/',
        },
)
