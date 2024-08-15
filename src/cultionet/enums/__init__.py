import enum


class StrEnum(str, enum.Enum):
    """
    Source:
        https://github.com/irgeek/StrEnum/blob/master/strenum/__init__.py
    """

    def __new__(cls, value, *args, **kwargs):
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return self.value


class DataColumns(StrEnum):
    GEOID = "geo_id"
    YEAR = "year"


class AttentionTypes(StrEnum):
    FRACTAL = "fractal"
    NATTEN = "natten"
    SPATIAL_CHANNEL = "spatial_channel"


class CLISteps(StrEnum):
    CREATE = "create"
    CREATE_PREDICT = "create-predict"
    GRAPH = "graph"
    SKFOLDCV = "skfoldcv"
    TRAIN = "train"
    TRAIN_TRANSFER = "train-transfer"
    PREDICT = "predict"
    PREDICT_TRANSFER = "predict-transfer"
    VERSION = "version"


class Destinations(StrEnum):
    CKPT = 'ckpt'
    DATA = 'data'
    FIGURES = 'figures'
    PREDICT = 'predict'
    PROCESSED = 'processed'
    TRAIN = 'train'
    TEST = 'test'
    TIME_SERIES_VARS = 'time_series_vars'
    USER_TRAIN = 'user_train'


class LossTypes(StrEnum):
    CLASS_BALANCED_MSE = "ClassBalancedMSELoss"
    TANIMOTO_COMPLEMENT = "TanimotoComplementLoss"
    TANIMOTO = "TanimotoDistLoss"
    TANIMOTO_COMBINED = "TanimotoCombined"
    TOPOLOGY = "TopologyLoss"


class ModelNames(StrEnum):
    CLASS_INFO = "classes.info"
    CKPT_NAME = "last.ckpt"
    CKPT_TRANSFER_NAME = "last_transfer.ckpt"
    NORM = "last.norm"


class ModelTypes(StrEnum):
    TOWERUNET = 'TowerUNet'


class ResBlockTypes(StrEnum):
    RES = 'res'
    RESA = 'resa'


class LearningRateSchedulers(StrEnum):
    COSINE_ANNEALING_LR = 'CosineAnnealingLR'
    EXPONENTIAL_LR = 'ExponentialLR'
    ONE_CYCLE_LR = 'OneCycleLR'
    STEP_LR = 'StepLR'
