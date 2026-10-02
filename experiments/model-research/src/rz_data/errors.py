"""Errors at the external data boundary."""


class DataError(ValueError):
    """A typed, contextual rejection; never a replacement label."""

    def __init__(self, code: str, context: str, message: str):
        self.code = code
        self.context = context
        self.message = message
        super().__init__(f"{code} at {context}: {message}")

    def as_dict(self) -> dict:
        return {"code": self.code, "context": self.context, "message": self.message}
