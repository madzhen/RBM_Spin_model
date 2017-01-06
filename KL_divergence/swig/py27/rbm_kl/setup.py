"""
setup.py file for SWIG rbm_kl
"""

from setuptools import setup, Extension


rbm_kl_module = Extension('_rbm_kl',
                         sources = ['rbm_kl_wrap.cxx','rbm_kl.cpp'],
                         )

setup (name = 'rbm_kl',
       version = '0.1',
       author = 'Zhenyu',
       description = """rbm_kl swig""",
       ext_modules = [rbm_kl_module],
       py_modules = ["rbm_kl"],
       license = "GPL",
       keywords = ("rbm_kl", "rbm_egg"),
       platforms = "Independant",
       url = "",
       install_requires=None,
       )