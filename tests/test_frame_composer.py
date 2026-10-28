"""Fixed even canvas, centering, and the error frame."""

import numpy as np
import pytest

from projection.frame_composer import FrameComposer, even_ceil


def test_even_ceil_and_centered_image():
    assert even_ceil(1) == 2
    assert even_ceil(448) == 448
    assert even_ceil(401) == 402
    composer = FrameComposer(10, 8)
    image = np.full((4, 4), 255, dtype=np.uint8)
    frame = composer.compose(image)
    assert frame.shape == (10, 8)
    assert frame.sum() == 255 * 16
    assert frame[3, 2] == 255
    assert frame[0, 0] == 0
    with pytest.raises(ValueError):
        FrameComposer(5, 4)


def test_error_frame_matches_canvas_and_is_not_blank():
    composer = FrameComposer(80, 120)
    frame = composer.error_frame("Row 3: invalid plane — |ab|-|ac| = 20.5288 >= EPS")
    assert frame.shape == (80, 120)
    assert frame.max() > 0
    assert frame[0, 0] == 0
