from typing import Optional


class Text:
    def __init__(self, original: str, transformed: Optional[str] = None):
        self.original: str = original
        self.transformed: str = transformed if transformed else self.original

    def __str__(self):
        return (
            f"Text:\n" f"original={self.original}\n" f"transformed={self.transformed}"
        )
