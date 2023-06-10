import enum


class CLISteps(enum.Enum):
    CREATE = enum.auto()
    CREATE_PREDICT = "create-predict"
    GRAPH = enum.auto()
    SKFOLDCV = enum.auto()
    TRAIN = enum.auto()
    TRAIN_TRANSFER = "train-transfer"
    PREDICT = enum.auto()
    PREDICT_TRANSFER = "predict-transfer"
    VERSION = enum.auto()


class Destinations(enum.Enum):
    CKPT = enum.auto()
    DATA = enum.auto()
    FIGURES = enum.auto()
    PREDICT = enum.auto()
    PROCESSED = enum.auto()
    TRAIN = enum.auto()
    TEST = enum.auto()
    TIME_SERIES_VARS = enum.auto()
    USER_TRAIN = enum.auto()


class ModelNames(enum.Enum):
    CLASS_INFO = "classes.info"
    CKPT_NAME = "last.ckpt"
    CKPT_TRANSFER_NAME = "last_transfer.ckpt"
    NORM = "last.norm"


class ModelTypes(enum.Enum):
    UNET = enum.auto()
    RESUNET = enum.auto()


class ResBlockTypes(enum.Enum):
    RES = enum.auto()
    RESA = enum.auto()
