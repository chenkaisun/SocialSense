# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..file_utils import requires_backends


class ImageFeatureExtractionMixin:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class CLIPFeatureExtractor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class CLIPProcessor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["vision"])


class DeiTFeatureExtractor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class DetrFeatureExtractor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ViTFeatureExtractor:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
