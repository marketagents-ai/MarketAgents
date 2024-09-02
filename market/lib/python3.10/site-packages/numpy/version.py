
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.1.0"
__version__ = version
full_version = version

git_revision = "2f7fe64b8b6d7591dd208942f1cc74473d5db4cb"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
