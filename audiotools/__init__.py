# Code borrowed from https://github.com/keras-team/keras
# License: Apache License 2.0

from audiotools import backend

if backend.backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

from audiotools.api_export import audiotools_export

if backend.backend() == "torch":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "jax":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "tensorflow":
    raise RuntimeError(f"Invalid backend: {backend.backend()}")
    # backend_name_scope = backend.tensorflow.core.name_scope
elif backend.backend() == "numpy":
    raise RuntimeError(f"Invalid backend: {backend.backend()}")
    # from audiotools.backend.numpy.core import Variable as NumpyVariable
    # backend_name_scope = backend.common.name_scope.name_scope
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")

# Import backend functions.
if backend.backend() == "torch":
    from audiotools.backend.torch import *  # noqa: F403

    distribution_lib = None
elif backend.backend() == "jax":
    from audiotools.backend.jax import *  # noqa: F403
elif backend.backend() == "tensorflow":
    raise ValueError("Audiotools does not have a tensorflow backend in development yet.")
    # from audiotools.backend.tensorflow import *  # noqa: F403
elif backend.backend() == "numpy":
    raise ValueError("Audiotools does not have a numpy backend in development yet.")
    # from audiotools.backend.numpy import *  # noqa: F403

    distribution_lib = None
else:
    raise ValueError(f"Unable to import backend : {backend()}")


@audiotools_export("audiotools.name_scope")
class name_scope(backend_name_scope):
    pass


@audiotools_export("audiotools.device")
def device(device_name):
    return backend.device_scope(device_name)

# from audiotools import activations
# from audiotools import applications
# from audiotools import backend
# from audiotools import constraints
# from audiotools import datasets
# from audiotools import initializers
# from audiotools import layers
# from audiotools import models
# from audiotools import ops
# from audiotools import optimizers
# from audiotools import regularizers
# from audiotools import utils
# from audiotools.backend import KerasTensor
# from audiotools.layers import Input
# from audiotools.layers import Layer
# from audiotools.models import Functional
# from audiotools.models import Model
# from audiotools.models import Sequential
from audiotools.version import __version__
