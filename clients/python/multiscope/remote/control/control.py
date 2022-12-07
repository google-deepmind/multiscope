"""Utilities for changing multiscope behaviour conditionally."""

from absl import flags

DISABLE_MULTISCOPE = flags.DEFINE_bool(
    "multiscope_disable",
    False,
    "Globally disable multiscope and cause its calls to become no-ops.",
)


# Module local variable that controls whether all calls to multiscope are
# no-ops. Useful for conditionally disabling multiscope, eg. in tests.
_disabled = False


def enable():
    """If multiscope is disabled, this re-enables it."""
    global _disabled
    _disabled = False


def disable():
    """All multiscope calls become no-ops."""
    global _disabled
    _disabled = True


def disabled():
    """Returns true if multiscope is disabled."""
    return _disabled or DISABLE_MULTISCOPE.value
