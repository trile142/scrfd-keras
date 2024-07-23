from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LayerNormalization, ReLU, MaxPooling2D, AveragePooling2D, Add
from tensorflow_addons.layers import GroupNormalization

from mmdet_keras.utils import make_layer


def _build_norm_layer(x,
                      norm_cfg,
                      name="",
                      prefix="",
                      postfix="",
                      layers_dict={}):
    if norm_cfg is not None:
        norm_type = norm_cfg['type']
        if name == "":
            key_name = norm_type.lower() + str(postfix)
            name = f"{prefix}.{key_name}"
        if norm_type == 'BN':
            x = make_layer(BatchNormalization(epsilon=1e-5, name=name), layers_dict)(x)
        elif norm_type == 'GN':
            x = make_layer(GroupNormalization(groups=norm_cfg['num_groups'], epsilon=1e-5, name=name), layers_dict)(x)
        elif norm_type == 'LN':
            x = make_layer(LayerNormalization(epsilon=1e-5, name=name), layers_dict)(x)
    return x


def _stem(input,
          in_channels,
          stem_channels,
          norm_cfg,
          prefix="",
          layers_dict={}):
    # self.stem = nn.Sequential(
    # build_conv_layer(
    #     self.conv_cfg,
    #     in_channels,
    #     stem_channels // 2,
    #     kernel_size=3,
    #     stride=2,
    #     padding=1,
    #     bias=False),
    x = ZeroPadding2D(1)(input)
    x = make_layer(Conv2D(stem_channels // 2, 3, strides=2, use_bias=False, name=f"{prefix}.0"), layers_dict)(x)
    # build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
    x = _build_norm_layer(x, norm_cfg, name=f"{prefix}.1", layers_dict=layers_dict)
    # nn.ReLU(inplace=True),
    x = ReLU()(x)
    # build_conv_layer(
    #     self.conv_cfg,
    #     stem_channels // 2,
    #     stem_channels // 2,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    #     bias=False),
    x = ZeroPadding2D(1)(x)
    x = make_layer(Conv2D(stem_channels // 2, 3, strides=1, use_bias=False, name=f"{prefix}.3"), layers_dict)(x)
    # build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
    x = _build_norm_layer(x, norm_cfg, name=f"{prefix}.4", layers_dict=layers_dict)
    # nn.ReLU(inplace=True),
    x = ReLU()(x)
    # build_conv_layer(
    #     self.conv_cfg,
    #     stem_channels // 2,
    #     stem_channels,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    #     bias=False),
    x = ZeroPadding2D(1)(x)
    x = make_layer(Conv2D(stem_channels, 3, strides=1, use_bias=False, name=f"{prefix}.6"), layers_dict)(x)
    # build_norm_layer(self.norm_cfg, stem_channels)[1],
    x = _build_norm_layer(x, norm_cfg, name=f"{prefix}.7", layers_dict=layers_dict)
    # nn.ReLU(inplace=True))
    x = ReLU()(x)
    return x


def _basic_block(x,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 prefix="",
                 layers_dict={}):
    assert dcn is None, 'Not implemented yet.'
    assert plugins is None, 'Not implemented yet.'

    # _inner_forward(x)
    identity = x

    # out = self.conv1(x)
    out = ZeroPadding2D(dilation)(x)
    out = make_layer(Conv2D(planes, 3, strides=stride, dilation_rate=dilation, use_bias=False, name=f"{prefix}.conv1"), layers_dict)(out)
    # out = self.norm1(out)
    out = _build_norm_layer(out, norm_cfg, prefix=prefix, postfix=1, layers_dict=layers_dict)
    # out = self.relu(out)
    out = ReLU()(out)
    # out = self.conv2(out)
    out = ZeroPadding2D(1)(out)
    out = make_layer(Conv2D(planes, 3, use_bias=False, name=f"{prefix}.conv2"), layers_dict)(out)
    # out = self.norm2(out)
    out = _build_norm_layer(out, norm_cfg, prefix=prefix, postfix=2, layers_dict=layers_dict)
    if downsample is not None:
        # identity = self.downsample(x)
        for i, layer in enumerate(downsample):
            layer_type = type(layer).__name__
            if layer_type == "AveragePooling2D":
                identity = layer(identity)
                continue
            layer._name = f"{prefix}.downsample.{i}"
            identity = make_layer(layer, layers_dict)(identity)
    # out += identity
    out = Add()([out, identity])
    # out = self.relu(out)
    out = ReLU()(out)
    return out


def _bottleneck(x,
                inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=None,
                style='pytorch',
                with_cp=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                dcn=None,
                plugins=None,
                prefix="",
                layers_dict={}):
    expansion = 4
    assert style in ['pytorch', 'caffe']
    assert dcn is None or isinstance(dcn, dict)
    assert plugins is None or isinstance(plugins, list)
    if plugins is not None:
        allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
        assert all(p['position'] in allowed_position for p in plugins)

    # with_dcn = dcn is not None
    # with_plugins = plugins is not None

    if style == 'pytorch':
        conv1_stride = 1
        conv2_stride = stride
    else:
        conv1_stride = stride
        conv2_stride = 1

    # _inner_forward(x)
    identity = x
    # out = self.conv1(x)
    out = make_layer(Conv2D(planes, 1, strides=conv1_stride, use_bias=False, name=f"{prefix}.conv1"), layers_dict)(x)
    # out = self.norm1(out)
    out = _build_norm_layer(out, norm_cfg, prefix=prefix, postfix=1, layers_dict=layers_dict)
    # out = self.relu(out)
    out = ReLU()(out)

    # if self.with_plugins:
    #     out = self.forward_plugin(out, self.after_conv1_plugin_names)

    # out = self.conv2(out)
    out = ZeroPadding2D(dilation)(out)
    out = make_layer(Conv2D(planes, 3, strides=conv2_stride, dilation_rate=dilation, use_bias=False, name=f"{prefix}.conv2"), layers_dict)(out)
    # out = self.norm2(out)
    out = _build_norm_layer(out, norm_cfg, prefix=prefix, postfix=2, layers_dict=layers_dict)
    # out = self.relu(out)
    out = ReLU()(out)

    # if self.with_plugins:
    #     out = self.forward_plugin(out, self.after_conv2_plugin_names)

    # out = self.conv3(out)
    out = make_layer(Conv2D(planes * expansion, 1, use_bias=False, name=f"{prefix}.conv3"), layers_dict)(out)
    # out = self.norm3(out)
    out = _build_norm_layer(out, norm_cfg, prefix=prefix, postfix=3, layers_dict=layers_dict)

    # if self.with_plugins:
    #     out = self.forward_plugin(out, self.after_conv3_plugin_names)

    if downsample is not None:
        # identity = self.downsample(x)
        for i, layer in enumerate(downsample):
            layer_type = type(layer).__name__
            if layer_type == "AveragePooling2D":
                identity = layer(identity)
                continue
            layer._name = f"{prefix}.downsample.{i}"
            identity = make_layer(layer, layers_dict)(identity)

    # out += identity
    out = Add()([out, identity])

    # out = self.relu(out)
    out = ReLU()(out)

    return out


arch_settings = {
    0: (_basic_block, (2, 2, 2, 2)),
    18: (_basic_block, (2, 2, 2, 2)),
    19: (_basic_block, (2, 4, 4, 1)),
    20: (_basic_block, (2, 3, 2, 2)),
    22: (_basic_block, (2, 4, 3, 1)),
    24: (_basic_block, (2, 4, 4, 1)),
    26: (_basic_block, (2, 4, 4, 2)),
    28: (_basic_block, (2, 5, 4, 2)),
    29: (_basic_block, (2, 6, 3, 2)),
    30: (_basic_block, (2, 5, 5, 2)),
    32: (_basic_block, (2, 6, 5, 2)),
    34: (_basic_block, (3, 4, 6, 3)),
    35: (_basic_block, (3, 6, 4, 3)),
    38: (_basic_block, (3, 8, 4, 3)),
    40: (_basic_block, (3, 8, 5, 3)),
    50: (_bottleneck, (3, 4, 6, 3)),
    56: (_bottleneck, (3, 8, 4, 3)),
    68: (_bottleneck, (3, 10, 6, 3)),
    74: (_bottleneck, (3, 12, 6, 3)),
    101: (_bottleneck, (3, 4, 23, 3)),
    152: (_bottleneck, (3, 8, 36, 3))
}


def _make_stage_plugins(plugins, num_stages, stage_idx):
    """Make plugins for ResNet ``stage_idx`` th stage.

    Currently we support to insert ``context_block``,
    ``empirical_attention_block``, ``nonlocal_block`` into the backbone
    like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
    Bottleneck.

    An example of plugins format could be:

    Examples:
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

    Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

    .. code-block:: none

        conv1-> conv2->conv3->yyy->zzz1->zzz2

    Suppose 'stage_idx=1', the structure of blocks in the stage would be:

    .. code-block:: none

        conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

    If stages is missing, the plugin would be applied to all stages.

    Args:
        plugins (list[dict]): List of plugins cfg to build. The postfix is
            required if multiple same type plugins are inserted.
        stage_idx (int): Index of stage to build

    Returns:
        list[dict]: Plugins for current stage
    """
    stage_plugins = []
    for plugin in plugins:
        plugin = plugin.copy()
        stages = plugin.pop('stages', None)
        assert stages is None or len(stages) == num_stages
        # whether to insert plugin into current stage
        if stages is None or stages[stage_idx]:
            stage_plugins.append(plugin)

    return stage_plugins


def _res_layer(input,
               block,
               block_expansion,
               inplanes,
               planes,
               num_blocks,
               stride=1,
               avg_down=False,
               conv_cfg=None,
               norm_cfg=dict(type='BN'),
               downsample_first=True,
               prefix="",
               layers_dict={},
               **kwargs):
    downsample = None
    if stride != 1 or inplanes != planes * block_expansion:
        downsample = []
        conv_stride = stride
        if avg_down:
            conv_stride = 1
            downsample.append(
                AveragePooling2D(
                    pool_size=stride,
                    strides=stride,
                    padding='same'
                ))
        downsample.append(
            Conv2D(planes * block_expansion, kernel_size=1, strides=conv_stride, use_bias=False),
        )
        if norm_cfg is not None:
            norm_type = norm_cfg['type']
            layer = None
            if norm_type == 'BN':
                layer = BatchNormalization(epsilon=1e-5)
            elif norm_type == 'GN':
                layer = GroupNormalization(groups=norm_cfg['num_groups'], epsilon=1e-5)
            elif norm_type == 'LN':
                layer = LayerNormalization(epsilon=1e-5)
            downsample.append(layer)

    x = input
    if downsample_first:
        x = block(x,
                  inplanes=inplanes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  conv_cfg=conv_cfg,
                  norm_cfg=norm_cfg,
                  prefix=f"{prefix}.0",
                  layers_dict=layers_dict,
                  **kwargs)
        inplanes = planes * block_expansion
        for i in range(1, num_blocks):
            x = block(x,
                      inplanes=inplanes,
                      planes=planes,
                      stride=1,
                      conv_cfg=conv_cfg,
                      norm_cfg=norm_cfg,
                      prefix=f"{prefix}.{i}",
                      layers_dict=layers_dict,
                      **kwargs)
    else:
        for i in range(num_blocks - 1):
            x = block(x,
                      inplanes=inplanes,
                      planes=planes,
                      stride=1,
                      conv_cfg=conv_cfg,
                      norm_cfg=norm_cfg,
                      prefix=f"{prefix}.{i}",
                      layers_dict=layers_dict,
                      **kwargs)
        x = block(x,
                  inplanes=inplanes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  conv_cfg=conv_cfg,
                  norm_cfg=norm_cfg,
                  prefix=f"{prefix}.{num_blocks-1}",
                  layers_dict=layers_dict,
                  **kwargs)
    return x


def _resnet(input,
            depth,
            in_channels=3,
            stem_channels=None,
            base_channels=64,
            num_stages=4,
            block_cfg=None,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3),
            style='pytorch',
            deep_stem=False,
            avg_down=False,
            no_pool33=False,
            frozen_stages=-1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            dcn=None,
            stage_with_dcn=(False, False, False, False),
            plugins=None,
            with_cp=False,
            zero_init_residual=True,
            prefix="",
            layers_dict={}):
    # init parameters
    if depth not in arch_settings:
        raise KeyError(f'invalid depth {depth} for resnet')
    if stem_channels is None:
        stem_channels = base_channels
    assert num_stages >= 1 and num_stages <= 4
    assert len(strides) == len(dilations) == num_stages
    assert max(out_indices) < num_stages
    if dcn is not None:
        assert len(stage_with_dcn) == num_stages
    if block_cfg is None:
        block, stage_blocks = arch_settings[depth]
    else:
        block = _basic_block if block_cfg['block'] == 'BasicBlock' else _bottleneck
        stage_blocks = block_cfg['stage_blocks']
        assert len(stage_blocks) >= num_stages
    if block == _basic_block:
        block_expansion = 1
    else:
        block_expansion = 4
    stage_blocks = stage_blocks[:num_stages]
    inplanes = stem_channels

    # self._make_stem_layer(in_channels, stem_channels)
    x = input
    if deep_stem:
        # x = self.stem(x)
        x = _stem(x, in_channels, stem_channels, norm_cfg, prefix=f"{prefix}.stem", layers_dict=layers_dict)
    else:
        pass
        # x = self.conv1(x)
        x = ZeroPadding2D(3)(x)
        x = make_layer(Conv2D(stem_channels, 7, strides=2, use_bias=False, name=f"{prefix}.conv1"), layers_dict)(x)
        # x = self.norm1(x)
        x = _build_norm_layer(x, norm_cfg, name=f"{prefix}.norm1", layers_dict=layers_dict)
        # x = self.relu(x)
        x = ReLU()(x)
    # x = self.maxpool(x)
    if no_pool33:
        assert deep_stem
        x = MaxPooling2D(2, strides=2)(x)
    else:
        x = ZeroPadding2D(1)(x)
        x = MaxPooling2D(3, strides=2)(x)

    if block_cfg is not None and 'stage_planes' in block_cfg:
        stage_planes = block_cfg['stage_planes']
    else:
        stage_planes = [base_channels * 2**i for i in range(num_stages)]

    outs = []
    for i, num_blocks in enumerate(stage_blocks):
        stride = strides[i]
        dilation = dilations[i]
        dcn = dcn if stage_with_dcn[i] else None
        if plugins is not None:
            stage_plugins = _make_stage_plugins(plugins, num_stages, i)
        else:
            stage_plugins = None
        planes = stage_planes[i]
        x = _res_layer(x,
                       block=block,
                       block_expansion=block_expansion,
                       inplanes=inplanes,
                       planes=planes,
                       num_blocks=num_blocks,
                       stride=stride,
                       dilation=dilation,
                       style=style,
                       avg_down=avg_down,
                       with_cp=with_cp,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       dcn=dcn,
                       plugins=stage_plugins,
                       prefix=f"{prefix}.layer{i+1}",
                       layers_dict=layers_dict)
        if i in out_indices:
            outs.append(x)
        inplanes = planes * block_expansion

    return outs


def _resnetv1e(input,
               prefix="",
               layers_dict={},
               **kwargs):
    return _resnet(input=input,
                   deep_stem=True,
                   avg_down=True,
                   no_pool33=True,
                   **kwargs,
                   prefix=prefix,
                   layers_dict=layers_dict)
