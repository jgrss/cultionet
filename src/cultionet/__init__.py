__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '1.4.0'
from .model import fit, load_model, predict

__all__ = ['fit', 'fit_maskrcnn', 'load_model', 'predict']
