# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy
from setuptools import setup, find_packages
# import numpy.distutils.misc_util

# ext_modules = [Extension(
#        "nearest_neighbors",
#        sources=["knn.pyx", "knn_.cxx",],  # source file(s)
#        include_dirs=["./", numpy.get_include()],
#        language="c++",            
#        extra_compile_args = [ "-std=c++11", "-fopenmp",],
#        extra_link_args=["-std=c++11", '-fopenmp'],
#   )]

# setup(
#     name = "KNN NanoFLANN",
#     ext_modules = ext_modules,
#     cmdclass = {'build_ext': build_ext},
# )

# m_name = "grid_subsampling"

# SOURCES = ["../cpp_utils/cloud/cloud.cpp",
#            "grid_subsampling/grid_subsampling.cpp",
#            "wrapper.cpp"]

# module = Extension(m_name,
#                    sources=SOURCES,
#                    extra_compile_args=['-std=c++11',
#                                        '-D_GLIBCXX_USE_CXX11_ABI=0'])

# setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())

setup(name='det3d',
      version='0.0.1',
      # list folders, not files
      packages=find_packages(exclude=['det3d.tests']),
      install_requires=[
          'shapely',
          'numpy',
          'easydict'
      ]
    )