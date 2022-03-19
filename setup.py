from setuptools import setup, find_packages

__version__ = None  # This will get replaced when reading version.py
exec(open('haxballgym_tools/version.py').read())

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='haxballgym_tools',
    packages=find_packages(),
    version=__version__,
    description='Extra tools for HaxballGym, like SB3 compatibility',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wazarr',
    install_requires=[
        'haxballgym>=0.3.0',
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['haxball', 'gym', 'reinforcement-learning'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
)