__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '1.3.3'
from .model import fit, fit_maskrcnn, load_model, predict

__all__ = ['fit', 'fit_maskrcnn', 'load_model', 'predict']
