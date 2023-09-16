from keras_core import backend
import keras_core as keras
from focalnet_keras_core.layers import FocalModulation
import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("in_height", [8])
@pytest.mark.parametrize("in_width", [8])
@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("focal_window", [2, 3])
@pytest.mark.parametrize("focal_level", [3, 2])
@pytest.mark.parametrize("focal_factor", [2, 3])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("use_postln_in_modulation", [True, False])
@pytest.mark.parametrize("normalize_modulator", [True, False])
@pytest.mark.parametrize("proj_drop", [0.1, 0.5])
def test_focal_modulation_layer(
    batch_size,
    in_height,
    in_width,
    in_channels,
    dim,
    focal_window,
    focal_level,
    focal_factor,
    bias,
    proj_drop,
    use_postln_in_modulation,
    normalize_modulator,
):
    layer = FocalModulation(
        dim,
        focal_window,
        focal_level,
        focal_factor=focal_factor,
        bias=bias,
        proj_drop=proj_drop,
        use_postln_in_modulation=use_postln_in_modulation,
        normalize_modulator=normalize_modulator,
    )

    test_shapes = (batch_size, in_height, in_width, in_channels)

    x = keras.random.normal(test_shapes)

    # Check that forward pass works
    y = layer(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1]
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    if backend.backend() == "torch":
        assert y.requires_grad

        # Check that gradients are propagated
        y.sum().backward()

