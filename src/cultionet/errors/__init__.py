class TopologyClipError(Exception):
    """Raised when GeoPandas clip() fails because of topology errors"""
    pass


class TensorShapeError(Exception):
    """Raised when tensor shapes do not match"""
    pass
