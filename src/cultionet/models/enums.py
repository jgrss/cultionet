import enum


class ModelTypes(enum.StrEnum):
    UNET = enum.auto()
    RESUNET = enum.auto()


class ResBlockTypes(enum.StrEnum):
    RES = enum.auto()
    RESA = enum.auto()
