from . import detection, recognition, tools, data_generation, pipeline

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
