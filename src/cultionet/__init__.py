__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '1.6.1'
from .model import fit, load_model, predict, predict_lightning

__all__ = [
    'fit', 'fit_maskrcnn', 'load_model', 'predict', 'predict_lightning'
]
