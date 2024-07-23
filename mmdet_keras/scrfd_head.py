from tensorflow.keras.layers import Conv2D, BatchNormalization, LayerNormalization, ReLU, Layer, ZeroPadding2D, Reshape, Activation, DepthwiseConv2D, Lambda
from tensorflow_addons.layers import GroupNormalization

from mmdet_keras.anchor_generator import AnchorGenerator
from mmdet_keras.utils import make_layer


class Scale(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(name='scale', shape=(), initializer='ones')

    def call(self, x):
        return x * self.scale


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


def _depthwise_separable_conv_module(input,
                                     out_channels,
                                     kernel_size,
                                     padding=0,
                                     norm_cfg=None,
                                     act_cfg=dict(type='ReLU'),
                                     prefix="",
                                     layers_dict={}):
    in_channels = input.shape[-1]
    output = _conv_module(input, in_channels, kernel_size, padding=padding, groups=in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.depthwise_conv", layers_dict=layers_dict)
    output = _conv_module(output, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix=f"{prefix}.pointwise_conv", layers_dict=layers_dict)
    return output


def _get_conv_module(input,
                     out_channel,
                     dw_conv,
                     norm_cfg,
                     prefix="",
                     layers_dict={}):
    if not dw_conv:
        output = _conv_module(input, out_channel, 3, padding=1, norm_cfg=norm_cfg, prefix=prefix, layers_dict=layers_dict)
    else:
        output = _depthwise_separable_conv_module(
                input,
                out_channel,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                prefix=prefix,
                layers_dict=layers_dict)
    return output


def _scrfd_head(inputs,
                num_classes,
                in_channels,
                stacked_convs=4,
                feat_mults=None,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                loss_dfl=None,
                reg_max=8,
                cls_reg_share=False,
                strides_share=True,
                scale_mode = 1,
                dw_conv = False,
                use_kps = False,
                loss_kps=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.1),
                #loss_kps=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.3),
                # super params
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8, 16, 32],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                train_cfg=None,
                test_cfg=None,
                prefix="",
                layers_dict={}):

    use_dfl = True
    NK = 5
    if loss_dfl is None or not loss_dfl:
        use_dfl = False
    use_scale = False
    if scale_mode > 0 and (strides_share or scale_mode == 2):
        use_scale = True
    # print('USE-SCALE:', self.use_scale)

    ##### super
    use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
    # TODO better way to determine whether sample or not
    if use_sigmoid_cls:
        cls_out_channels = num_classes
    else:
        cls_out_channels = num_classes + 1

    if cls_out_channels <= 0:
        raise ValueError(f'num_classes={num_classes} is too small')

    _anchor_generator = AnchorGenerator(
        anchor_generator['strides'], anchor_generator['ratios'],
        anchor_generator['scales'], anchor_generator['base_sizes']
    )

    num_anchors = _anchor_generator.num_base_anchors[0]

    cls_scores = []
    bbox_preds = []
    kps_preds = []
    for idx, (x, stride) in enumerate(zip(inputs, _anchor_generator.strides)):
        # print('forward_single in stride:', stride)
        if strides_share:
            key = "0"
        else:
            key = str(stride).replace("(", "").replace(")", "").replace(", ", "_")
        # forward_single(x, scale, stride)
        cls_feat = x
        # cls_convs = cls_stride_convs[str(stride)]
        for i in range(stacked_convs):
            cls_feat = _get_conv_module(cls_feat, feat_channels, dw_conv, norm_cfg, prefix=f"{prefix}.cls_stride_convs.{key}.{i}", layers_dict=layers_dict)
        reg_feat = cls_feat
        # cls_score = cls_pred_module(cls_feat)
        cls_score = ZeroPadding2D(1)(cls_feat)
        cls_score = make_layer(Conv2D(cls_out_channels * num_anchors, 3, name=f"{prefix}.stride_cls.{key}"), layers_dict)(cls_score)
        # _bbox_pred = reg_pred_module(reg_feat)
        bbox_pred = ZeroPadding2D(1)(reg_feat)
        if not use_dfl:
            bbox_pred = make_layer(Conv2D(4 * num_anchors, 3, name=f"{prefix}.stride_reg.{key}"), layers_dict)(bbox_pred)
        else:
            bbox_pred = make_layer(Conv2D(4 * (reg_max + 1) * num_anchors, 3, name=f"{prefix}.stride_reg.{key}"), layers_dict)(bbox_pred)
        if use_scale:
            bbox_pred = make_layer(Scale(name=f"{prefix}.scales.{idx}"), layers_dict)(bbox_pred)
        if use_kps:
            # kps_pred = kps_pred_module(reg_feat)
            kps_pred = ZeroPadding2D(1)(reg_feat)
            kps_pred = make_layer(Conv2D(NK*2*num_anchors, 3, name=f"{prefix}.stride_kps.{key}"), layers_dict)(kps_pred)
        else:
            kps_pred = Lambda(lambda x: x[..., :NK*2])(reg_feat)
        # export to onnx
        cls_score = Reshape((-1, cls_out_channels))(cls_score)
        cls_score = Activation('sigmoid')(cls_score)
        bbox_pred = Reshape((-1, 4))(bbox_pred)
        kps_pred = Reshape((-1, 10))(kps_pred)

        cls_scores.append(cls_score)
        bbox_preds.append(bbox_pred)
        kps_preds.append(kps_pred)

    outputs = []
    outputs.extend(cls_scores)
    outputs.extend(bbox_preds)
    if use_kps:
        outputs.extend(kps_preds)
    return outputs
