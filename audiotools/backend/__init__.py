# Code borrowed from https://github.com/keras-team/keras
# License: Apache License 2.0

from audiotools.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

# from audiotools.backend.common.dtypes import result_type
# from audiotools.backend.common.audiotools_tensor import audiotoolsTensor
# from audiotools.backend.common.audiotools_tensor import any_symbolic_tensors
# from audiotools.backend.common.audiotools_tensor import is_audiotools_tensor
from audiotools.backend.common.name_scope import name_scope
# from audiotools.backend.common.stateless_scope import StatelessScope
# from audiotools.backend.common.stateless_scope import get_stateless_scope
# from audiotools.backend.common.stateless_scope import in_stateless_scope
# from audiotools.backend.common.variables import AutocastScope
# from audiotools.backend.common.variables import get_autocast_scope
# from audiotools.backend.common.variables import is_float_dtype
# from audiotools.backend.common.variables import is_int_dtype
# from audiotools.backend.common.variables import standardize_dtype
# from audiotools.backend.common.variables import standardize_shape
from audiotools.backend.config import epsilon
from audiotools.backend.config import floatx
from audiotools.backend.config import image_data_format
from audiotools.backend.config import set_epsilon
from audiotools.backend.config import set_floatx
from audiotools.backend.config import set_image_data_format
from audiotools.backend.config import standardize_data_format
