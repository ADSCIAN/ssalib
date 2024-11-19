"""Exceptions for SingularSpectrumAnalysis"""


class DecompositionError(Exception):
    """Exception raised when a dependent method is called before decompose"""

    def __init__(
            self,
            message: str
    ) -> None:
        super().__init__(message)


class ReconstructionError(Exception):
    """Exception raised when a dependent method is called before reconstruct"""

    def __init__(
            self,
            message: str
    ) -> None:
        super().__init__(message)
