from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, ZeroPadding2D, DepthwiseConv2D

from mmdet_keras.utils import make_layer


def _conv_bn(input,
             inp,
             oup,
             stride,
             prefix="",
             layers_dict={}):
    x = ZeroPadding2D(1)(input)
    x = make_layer(Conv2D(oup, 3, strides=stride, use_bias=False, name=f"{prefix}.0"), layers_dict)(x)
    x = make_layer(BatchNormalization(epsilon=1e-5, name=f"{prefix}.1"), layers_dict)(x)
    x = ReLU()(x)
    return x


def _conv_dw(input,
             inp,
             oup,
             stride,
             prefix="",
             layers_dict={}):
    x = ZeroPadding2D(1)(input)
    # x = make_layer(Conv2D(inp, 3, strides=stride, use_bias=False, groups=inp, name=f"{prefix}.0"), layers_dict)(x)
    x = make_layer(DepthwiseConv2D(3, strides=stride, use_bias=False, depth_multiplier=1, name=f"{prefix}.0"), layers_dict)(x)
    x = make_layer(BatchNormalization(epsilon=1e-5, name=f"{prefix}.1"), layers_dict)(x)
    x = ReLU()(x)

    x = make_layer(Conv2D(oup, 1, strides=1, use_bias=False, name=f"{prefix}.3"), layers_dict)(x)
    x = make_layer(BatchNormalization(epsilon=1e-5, name=f"{prefix}.4"), layers_dict)(x)
    x = ReLU()(x)
    return x


def _mobilenetv1(input,
                 in_channels=3,
                 block_cfg=None,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 prefix="",
                 layers_dict={}):
    if block_cfg is None:
        stage_planes = [8, 16, 32, 64, 128, 256]  # 0.25 default
        stage_blocks = [2, 4, 4, 2]
    else:
        stage_planes = block_cfg['stage_planes']
        stage_blocks = block_cfg['stage_blocks']
    assert len(stage_planes) == 6
    assert len(stage_blocks) == 4
    # x = self.stem(x)
    x = _conv_bn(input, 3, stage_planes[0], 2, prefix=f"{prefix}.stem.0", layers_dict=layers_dict)
    x = _conv_dw(x, stage_planes[0], stage_planes[1], 1, prefix=f"{prefix}.stem.1", layers_dict=layers_dict)
    # stage layers
    output = []
    for i, num_blocks in enumerate(stage_blocks):
        # x = stage_layer(x)
        for n in range(num_blocks):
            if n == 0:
                x = _conv_dw(x, stage_planes[i+1], stage_planes[i+2], 2, prefix=f"{prefix}.layer{i+1}.{n}", layers_dict=layers_dict)
            else:
                x = _conv_dw(x, stage_planes[i+2], stage_planes[i+2], 1, prefix=f"{prefix}.layer{i+1}.{n}", layers_dict=layers_dict)
        if i in out_indices:
            output.append(x)
    return output
