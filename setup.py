
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

from blazee import __version__


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass into py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        try:
            from multiprocessing import cpu_count
            self.pytest_args = ['-n', str(cpu_count()), '--boxed']
        except (ImportError, NotImplementedError):
            self.pytest_args = ['-n', '1', '--boxed']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


requires = [
    'numpy',
    'python-dateutil',
    'requests'
]

test_requirements = []

with open('README.md', 'r') as f:
    readme = f.read()
with open('HISTORY.md', 'r') as f:
    history = f.read()

setup(name='blazee',
      version=__version__,
      description='Blazee makes it easy to deploy Machine Learning models on the cloud and turn them into an awesome prediction API.',
      author='blazee.io',
      author_email='support@blazee.io',
      license='GNU General Public License v3.0',
      long_description=readme,
      long_description_content_type='text/markdown',
      package_data={'': ['LICENSE', 'NOTICE'], 'blazee': ['*.pem']},
      package_dir={'blazee': 'blazee'},
      include_package_data=True,
      url='https://github.com/blazee-io/blazee-python',
      packages=find_packages(),
      install_requires=requires,
      tests_require=test_requirements,
      cmdclass={'test': PyTest},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
      ])
