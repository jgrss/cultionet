from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

try:
    import numpy as np
except:
    raise ImportError('NumPy must be installed.')


def get_extensions():
    return [Extension('*', sources=['src/cultionet/networks/_build_network.pyx'])]


def setup_package():
    metadata = dict(
        ext_modules=cythonize(get_extensions()),
        include_dirs=[np.get_include()]
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
