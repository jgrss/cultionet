import enum


class ModelTypes(enum.Enum):
    UNET = enum.auto()
    RESUNET = enum.auto()


class ResBlockTypes(enum.Enum):
    RES = enum.auto()
    RESA = enum.auto()
