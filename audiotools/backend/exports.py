# Code borrowed from https://github.com/keras-team/keras
# License: Apache License 2.0

from audiotools import backend
from audiotools.api_export import audiotools_export

if backend.backend() == "tensorflow":
    raise RuntimeError(f"Invalid backend: {backend.backend()}")
    # BackendVariable = backend.tensorflow.core.Variable
    # backend_name_scope = backend.tensorflow.core.name_scope
elif backend.backend() == "jax":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "torch":
    backend_name_scope = backend.common.name_scope.name_scope
elif backend.backend() == "numpy":
    raise RuntimeError(f"Invalid backend: {backend.backend()}")
    # from audiotools.backend.numpy.core import Variable as NumpyVariable
    #
    # BackendVariable = NumpyVariable
    # backend_name_scope = backend.common.name_scope.name_scope
else:
    raise RuntimeError(f"Invalid backend: {backend.backend()}")


@audiotools_export("audiotools.name_scope")
class name_scope(backend_name_scope):
    pass


@audiotools_export("audiotools.device")
def device(device_name):
    return backend.device_scope(device_name)
