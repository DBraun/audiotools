# Code borrowed from https://github.com/keras-team/keras
# License: Apache License 2.0

from audiotools.api_export import audiotools_export

# Unique source of truth for the version number.
__version__ = "0.8.0"


@audiotools_export("audiotools.version")
def version():
    return __version__