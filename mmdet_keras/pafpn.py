from tensorflow.keras.layers import Conv2D, UpSampling2D, Add, BatchNormalization, LayerNormalization, ReLU, ZeroPadding2D, DepthwiseConv2D
from tensorflow_addons.layers import GroupNormalization

import warnings

from mmdet_keras.utils import make_layer


def _conv_module(input,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 prefix="",
                 layers_dict={}):
    use_bias = norm_cfg is None
    output = input
    if padding > 0:
        output = ZeroPadding2D(padding)(output)
    if groups == input.shape[-1]:  # depthwise
        output = make_layer(DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=out_channels//groups, use_bias=use_bias, name=f"{prefix}.conv"), layers_dict)(output)
    else:
        output = make_layer(Conv2D(out_channels, kernel_size, strides=stride, groups=groups, use_bias=use_bias, name=f"{prefix}.conv"), layers_dict)(output)
    if norm_cfg is not None:
        norm_type = norm_cfg['type']
        name = f"{prefix}.{norm_type.lower()}"
        if norm_type == 'BN':
            output = make_layer(BatchNormalization(epsilon=1e-5, name=name), layers_dict)(output)
        elif norm_type == 'GN':
            output = make_layer(GroupNormalization(groups=norm_cfg['num_groups'], epsilon=1e-5, name=name), layers_dict)(output)
        elif norm_type == 'LN':
            output = make_layer(LayerNormalization(epsilon=1e-5, name=name), layers_dict)(output)
    if act_cfg is not None:
        act_type = act_cfg['type']
        if act_type == 'ReLU':
            output = ReLU()(output)
    return output


def _pafpn(inputs,
           in_channels,
           out_channels,
           num_outs,
           start_level=0,
           end_level=-1,
           add_extra_convs=False,
           extra_convs_on_inputs=True,
           relu_before_extra_convs=False,
           no_norm_on_lateral=False,
           conv_cfg=None,
           norm_cfg=None,
           act_cfg=None,
           prefix="",
           layers_dict={}):
    """Forward function."""
    assert len(inputs) == len(in_channels)

    num_ins = len(in_channels)
    if end_level == -1:
        backbone_end_level = num_ins
        assert num_outs >= num_ins - start_level
    else:
        # if end_level < inputs, no extra level is allowed
        backbone_end_level = end_level
        assert end_level <= len(in_channels)
        assert num_outs == end_level - start_level
    assert isinstance(add_extra_convs, (str, bool))
    if isinstance(add_extra_convs, str):
        # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
        assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
    elif add_extra_convs:  # True
        if extra_convs_on_inputs:
            # TODO: deprecate `extra_convs_on_inputs`
            warnings.simplefilter('once')
            warnings.warn(
                '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                'Please use "add_extra_convs"', DeprecationWarning)
            add_extra_convs = 'on_input'
        else:
            add_extra_convs = 'on_output'

    # build laterals
    # lateral_convs
    laterals = []
    for i in range(start_level, backbone_end_level):
        lateral_i = _conv_module(inputs[i], out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.lateral_convs.{i-start_level}", layers_dict=layers_dict)
        # out = Conv2D(out_channels, 1, name=f"{prefix}.lateral_convs.{i-start_level}")(inputs[i])
        laterals.append(lateral_i)

    # build top-down path
    used_backbone_levels = len(laterals)
    for i in range(used_backbone_levels - 1, 0, -1):
        prev_shape = laterals[i - 1].shape[1:3]
        curr_shape = laterals[i].shape[1:3]
        interpolate_i = UpSampling2D(size=(prev_shape[0] // curr_shape[0], prev_shape[1] // curr_shape[1]))(laterals[i])
        laterals[i - 1] = Add()([laterals[i - 1], interpolate_i])

    # build outputs
    # part 1: from original levels
    # fpn_convs
    inter_outs = []
    for i in range(used_backbone_levels):
        inter_out_i = _conv_module(laterals[i], out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.fpn_convs.{i}", layers_dict=layers_dict)
        # out = Conv2D(out_channels, 3, padding='same', name=f"{prefix}.fpn_convs.{i}")(laterals[i])
        inter_outs.append(inter_out_i)

    # part 2: add bottom-up path
    # downsample_convs
    for i in range(0, used_backbone_levels - 1):
        downsample_i = _conv_module(inter_outs[i], out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.downsample_convs.{i}", layers_dict=layers_dict)
        # downsample_i = Conv2D(out_channels, 3, strides=2, padding='same', name=f"{prefix}.downsample_convs.{i}")(inter_outs[i])
        inter_outs[i + 1] = Add()([inter_outs[i + 1], downsample_i])

    # pafpn_convs
    outs = []
    outs.append(inter_outs[0])
    for i in range(1, used_backbone_levels):
        pafpn_conv_i = _conv_module(inter_outs[i], out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.pafpn_convs.{i-1}", layers_dict=layers_dict)
        # out = Conv2D(out_channels, 3, padding='same', name=f"{prefix}.pafpn_convs.{i-1}")(inter_outs[i])
        outs.append(pafpn_conv_i)
    return outs
