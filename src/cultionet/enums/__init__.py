import enum


class CLISteps(enum.Enum):
    CREATE = 'create'
    CREATE_PREDICT = "create-predict"
    GRAPH = 'graph'
    SKFOLDCV = 'skfoldcv'
    TRAIN = 'train'
    TRAIN_TRANSFER = "train-transfer"
    PREDICT = 'predict'
    PREDICT_TRANSFER = "predict-transfer"
    VERSION = 'version'


class Destinations(enum.Enum):
    CKPT = 'ckpt'
    DATA = 'data'
    FIGURES = 'figures'
    PREDICT = 'predict'
    PROCESSED = 'processed'
    TRAIN = 'train'
    TEST = 'test'
    TIME_SERIES_VARS = 'time_series_vars'
    USER_TRAIN = 'user_train'


class ModelNames(enum.Enum):
    CLASS_INFO = "classes.info"
    CKPT_NAME = "last.ckpt"
    CKPT_TRANSFER_NAME = "last_transfer.ckpt"
    NORM = "last.norm"


class ModelTypes(enum.Enum):
    UNET = 'unet'
    RESUNET = 'resunet'


class ResBlockTypes(enum.Enum):
    RES = 'res'
    RESA = 'resa'
