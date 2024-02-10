# Code borrowed from https://github.com/keras-team/keras
# License: Apache License 2.0

from audiotools.backend.common import global_state


class name_scope:
    """Creates a sub-namespace for variable paths.

    Args:
        name: Name of the current scope (string).
        caller: Optional ID of a caller object (e.g. class instance).
        deduplicate: If `True`, if `caller` was passed,
            and the previous caller matches the current caller,
            and the previous name matches the current name,
            do not reenter a new namespace.
    """

    def __init__(self, name, caller=None, deduplicate=True):
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name}"
            )
        self.name = name
        self.caller = caller
        self.deduplicate = deduplicate
        self._pop_on_exit = False

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[], set_to_default=True
        )
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        return self

    def __exit__(self, *args, **kwargs):
        if self._pop_on_exit:
            name_scope_stack = global_state.get_global_attribute(
                "name_scope_stack"
            )
            name_scope_stack.pop()


def current_path():
    name_scope_stack = global_state.get_global_attribute("name_scope_stack")
    if name_scope_stack is None:
        return ""
    return "/".join(x.name for x in name_scope_stack)