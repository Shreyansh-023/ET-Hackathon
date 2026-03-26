class PipelineError(Exception):
    """Base error for all pipeline failures."""


class ValidationPipelineError(PipelineError):
    """Raised when payload/schema validation fails."""


class ProviderPipelineError(PipelineError):
    """Raised when external provider calls fail."""


class RenderPipelineError(PipelineError):
    """Raised when render stage processing fails."""


class IOPipelineError(PipelineError):
    """Raised when filesystem/network IO fails."""
