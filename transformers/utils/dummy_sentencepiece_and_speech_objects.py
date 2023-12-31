# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..file_utils import requires_backends


class Speech2TextProcessor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["sentencepiece", "speech"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["sentencepiece", "speech"])
