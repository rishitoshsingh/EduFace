import math
import typing as typ
from fractions import Fraction

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo  

_ALL_VIDEO_FORMATS = [GstVideo.VideoFormat.from_string(
    f.strip()) for f in GstVideo.VIDEO_FORMATS_ALL.strip('{ }').split(',')]


def has_flag(value: GstVideo.VideoFormatFlags,
             flag: GstVideo.VideoFormatFlags) -> bool:

    # in VideoFormatFlags each new value is 1 << 2**{0...8}
    return bool(value & (1 << max(1, math.ceil(math.log2(int(flag))))))


def _get_num_channels(fmt: GstVideo.VideoFormat) -> int:
    """
        -1: means complex format (YUV, ...)
    """
    frmt_info = GstVideo.VideoFormat.get_info(fmt)
    
    # temporal fix
    if fmt == GstVideo.VideoFormat.BGRX:
        return 4
    
    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.ALPHA):
        return 4

    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.RGB):
        return 3

    if has_flag(frmt_info.flags, GstVideo.VideoFormatFlags.GRAY):
        return 1

    return -1

_ALL_VIDEO_FORMAT_CHANNELS = {fmt: _get_num_channels(fmt) for fmt in _ALL_VIDEO_FORMATS}


def get_num_channels(fmt: GstVideo.VideoFormat):
    return _ALL_VIDEO_FORMAT_CHANNELS[fmt]


_DTYPES = {
    16: np.int16,
}


def get_np_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    format_info = GstVideo.VideoFormat.get_info(fmt)
    return _DTYPES.get(format_info.bits, np.uint8)