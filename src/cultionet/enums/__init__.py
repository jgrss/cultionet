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


class InferenceNames(StrEnum):
    CLASSES_L2 = 'classes_l2'
    CLASSES_L3 = 'classes_l3'
    CROP_TYPE = 'crop_type'
    DISTANCE = 'distance'
    EDGE = 'edge'
    CROP = 'crop'
    RECONSTRUCTION = 'reconstruction'


class LossTypes(StrEnum):
    BOUNDARY = "BoundaryLoss"
    CENTERLINE_DICE = "CLDiceLoss"
    CLASS_BALANCED_MSE = "ClassBalancedMSELoss"
    CROSS_ENTROPY = "CrossEntropyLoss"
    LOG_COSH = "LogCoshLoss"
    FOCAL_TVERSKY = "FocalTverskyLoss"
    TANIMOTO_COMPLEMENT = "TanimotoComplementLoss"
    TANIMOTO = "TanimotoDistLoss"
    TANIMOTO_COMBINED = "TanimotoCombined"
    TVERSKY = "TverskyLoss"


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


class ValidationNames(StrEnum):
    TRUE_CROP = 'true_crop'
    TRUE_EDGE = 'true_edge'
    TRUE_CROP_AND_EDGE = 'true_crop_and_edge'
    TRUE_CROP_OR_EDGE = 'true_crop_or_edge'
    TRUE_CROP_TYPE = 'true_crop_type'
    MASK = 'mask'  # 1|0 mask
