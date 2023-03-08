class TopologyClipError(Exception):
    """Raised when GeoPandas clip() fails because of topology errors."""

    pass


class TensorShapeError(Exception):
    """Raised when tensor shapes do not match."""

    def __init__(
        self, message: str = "The tensor shapes do not match."
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
