"""Errors that the command line prints without a traceback."""


class UserFacingError(Exception):
    """A problem in the user's input, with a message that is safe to print."""


class NrrdReadError(UserFacingError):
    """The NRRD path is missing, unreadable, or not a supported volume."""


class CsvFormatError(UserFacingError):
    """The CSV file is missing, unreadable, or has no data rows."""


class PipelineError(UserFacingError):
    """The run cannot produce a video (for example, every row was skipped)."""
