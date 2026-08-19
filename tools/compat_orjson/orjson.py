"""Small diagnostic-only orjson compatibility shim for the CUDA capsule.

The pinned ROCm environment supplies the binary orjson wheel through the
SGLang overlay.  The separate CUDA golden environment does not; the capsule
only needs JSON serialization while importing SGLang's FLA modules, so use the
stdlib implementation there and never install this as a product dependency.
"""

import json

JSONDecodeError = json.JSONDecodeError
OPT_APPEND_NEWLINE = 1 << 0
OPT_INDENT_2 = 1 << 1
OPT_NON_STR_KEYS = 1 << 2
OPT_SORT_KEYS = 1 << 3
OPT_SERIALIZE_NUMPY = 1 << 4
OPT_OMIT_MICROSECONDS = 1 << 5
OPT_NAIVE_UTC = 1 << 6
OPT_UTC_Z = 1 << 7
OPT_PASSTHROUGH_DATACLASS = 1 << 8
OPT_PASSTHROUGH_DATETIME = 1 << 9
OPT_PASSTHROUGH_SUBCLASS = 1 << 10
OPT_SERIALIZE_DATACLASS = 1 << 11
OPT_SERIALIZE_UUID = 1 << 12
OPT_FRAGMENT = 1 << 13
OPT_APPEND_NEWLINE = 1 << 14


def dumps(value, *, default=None, option=0):
    indent = 2 if option & OPT_INDENT_2 else None
    sort_keys = bool(option & OPT_SORT_KEYS)
    text = json.dumps(
        value,
        default=default,
        indent=indent,
        sort_keys=sort_keys,
        separators=None if indent else (",", ":"),
    )
    if option & OPT_APPEND_NEWLINE:
        text += "\n"
    return text.encode("utf-8")


def loads(value, /):
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    return json.loads(value)
