#!/usr/bin/env python
import os
import pkgutil
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from subprocess import check_call, CalledProcessError

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
relic.release.write_template(version, os.path.join(*PACKAGENAME.split('.')))


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


INSTALL_REQUIRES = [
    'numpy',
    'astropy>=3.1',
    'gwcs',
    'stsci.stimage',
    'stsci.imagestats',
    'spherical_geometry',
]
TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'codecov'
]
DOCS_REQUIRE = [
    'numpydoc'
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
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    ext_modules=[],
    extras_require={
        'docs': DOCS_REQUIRE,
        'test': TESTS_REQUIRE,
    },
    cmdclass={
        'test': PyTest,
    },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/tweakwcs/issues/',
        'Source': 'https://github.com/spacetelescope/tweakwcs/',
        'Help': 'https://hsthelp.stsci.edu/',
    },
)
