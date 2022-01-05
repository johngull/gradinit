import numpy as np
import torch
from torchvision.models import vgg11, resnet18, efficientnet_b0, regnet_x_8gf
from gradinit import GradInitWrapper

import typing

IMAGE_MODEL_LIST = [vgg11, resnet18, efficientnet_b0, regnet_x_8gf]


def _copy_module_parameters(module: torch.nn.Module) -> typing.Dict[str, torch.Tensor]:
    res = {}
    for name, parameter in module.named_parameters():
        res[name] = parameter.clone()
    return res


def _compare_two_parameters_dict_equal(
    params1: typing.Dict[str, torch.Tensor], params2: typing.Dict[str, torch.Tensor]
):
    # check names of parameters ar the same
    assert not (set(params1) - set(params2))
    assert not (set(params2) - set(params1))

    # check values
    for k, v in params1.items():
        assert torch.allclose(v, params2[k])


def _compare_two_parameters_dict_scaled(
    params1: typing.Dict[str, torch.Tensor], params2: typing.Dict[str, torch.Tensor]
):
    # check names of parameters ar the same
    assert not (set(params1) - set(params2))
    assert not (set(params2) - set(params1))

    # check values
    scales = []
    for k, v in params1.items():
        if torch.all(params2[k].abs() < 1e-6):
            scale = 1.0
        else:
            mask = params2[k].abs() > 1e-6
            scale = torch.mean(v[mask] / params2[k][mask])
        assert torch.allclose(v, scale * params2[k])
        scales.append(scale)

    scales = np.array(scales)
    # check that all scales bigger
    assert np.all(scales >= 0.01)

    # check that some of scales are not 1
    assert np.any(np.abs(scales - 1) > 1e-6)


def _check_model_equality_with_wrap_action(
    model: typing.Callable[[], torch.nn.Module],
    action: typing.Callable[[torch.nn.Module], None],
):
    model = model()
    params = _copy_module_parameters(model)
    action(model)
    new_params = _copy_module_parameters(model)
    _compare_two_parameters_dict_equal(params, new_params)


def _random_image_classification_gradinit(
    model: torch.nn.Module,
    steps: int = 10,
    batch_size: int = 4,
    input_sz: typing.Tuple = (224, 224),
    classes_count=1000,
):
    with GradInitWrapper(model) as ginit:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(steps):
            x = torch.randn(batch_size, 3, *input_sz)
            y = torch.randint(0, classes_count, (batch_size,))
            pred = model(x)

            loss = torch.nn.functional.cross_entropy(pred, y)
            loss = ginit.grad_loss(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_no_changes_on_wrap_detach():
    """
    Test for keeping model unchanged during wrapping and unwrapping using detach
    """

    def wrap_detach(model: torch.nn.Module):
        ginit = GradInitWrapper(model)
        ginit.detach()

    for model in IMAGE_MODEL_LIST:
        _check_model_equality_with_wrap_action(model, wrap_detach)


def test_no_changes_on_with_wrap():
    """
    Test for keeping model unchanged during wrapping and unwrapping using with-notation
    """

    def with_wrap(model: torch.nn.Module):
        with GradInitWrapper(model) as ginit:
            pass

    for model in IMAGE_MODEL_LIST:
        _check_model_equality_with_wrap_action(model, with_wrap)


def test_only_scale_changes_on_gradinit():
    """
    Tests that all parameters only rescaled after gradinit procedure
    """
    for model in IMAGE_MODEL_LIST:
        model = model()
        params = _copy_module_parameters(model)
        _random_image_classification_gradinit(model)
        new_params = _copy_module_parameters(model)
        _compare_two_parameters_dict_scaled(params, new_params)


if __name__ == "__main__":
    test_no_changes_on_wrap_detach()
