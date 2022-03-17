__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '0.1.0'
from .model import fit, predict

__all__ = ['fit', 'predict']
