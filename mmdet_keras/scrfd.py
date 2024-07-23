import argparse
import os.path as osp
import os
from datetime import datetime

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import cv2

from mmdet_keras.mobilenet import _mobilenetv1
from mmdet_keras.resnet import _resnetv1e
from mmdet_keras.pafpn import _pafpn
from mmdet_keras.scrfd_head import _scrfd_head
from mmdet_keras.utils import _file2dict


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self,
                 config,
                 input_shape=(640, 640, 3),
                 ) -> None:
        self.config = _file2dict(config).model
        self.build_model(input_shape)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars(input_shape)

    def build_model(self, input_shape):
        self.layers_dict = {}
        input = Input(shape=input_shape, name="image")
        # backbone
        backbone_cfg = self.config.backbone.copy()
        backbone_type = backbone_cfg.pop('type')
        if backbone_type == 'MobileNetV1':
            x = _mobilenetv1(input,
                             **backbone_cfg,
                             prefix="backbone",
                             layers_dict=self.layers_dict)
        elif backbone_type == 'ResNetV1e':
            x = _resnetv1e(input,
                           **backbone_cfg,
                           prefix="backbone",
                           layers_dict=self.layers_dict)
        # neck
        neck_cfg = self.config.neck.copy()
        neck_type = neck_cfg.pop('type')
        if neck_type == 'PAFPN':
            x = _pafpn(x,
                       **neck_cfg,
                       prefix="neck",
                       layers_dict=self.layers_dict)
        # head
        bbox_head_cfg = self.config.bbox_head.copy()
        bbox_head_type = bbox_head_cfg.pop('type')
        if bbox_head_type == 'SCRFDHead':
            x = _scrfd_head(x,
                            **bbox_head_cfg,
                            prefix="bbox_head",
                            layers_dict=self.layers_dict)
        self.model = Model(input, x)

    def get_torch_name(self, name):
        import re
        name_components = name.split(".")
        new_name = ""
        for comp in name_components:
            res = re.search("^[0-9]*_[0-9]*$", comp)
            if res:
                nums = comp.split("_")
                new_comp = str(tuple(int(num) for num in nums))
            else:
                new_comp = comp
            new_name += new_comp + "."
        return new_name[:-1]

    def update_weights(self, state_dict):
        for layer in self.model.layers:
            if len(layer.weights) == 0:
                continue
            layer_type = type(layer).__name__
            layer_name = layer.name
            torch_layer_name = self.get_torch_name(layer_name)
            # print(torch_layer_name, layer_type, [w.shape for w in layer.get_weights()])
            # print(torch_layer_name, layer_type, layer.get_weights())
            weights = []
            weight_key = f'{torch_layer_name}.weight'
            bias_key = f'{torch_layer_name}.bias'
            if layer_type == "Conv2D":
                weights = [state_dict[weight_key].transpose(2, 3, 1, 0)]
                if bias_key in state_dict:
                    weights.append(state_dict[bias_key])
            elif layer_type == "DepthwiseConv2D":
                pt_weight = state_dict[weight_key]
                w_shape = pt_weight.shape
                depth_multiplier = layer.depth_multiplier
                weights = [pt_weight.reshape(w_shape[0] // depth_multiplier, depth_multiplier, w_shape[2], w_shape[3]).transpose(2, 3, 0, 1)]
                if bias_key in state_dict:
                    weights.append(state_dict[bias_key])
            elif layer_type == "Dense":
                weights = [state_dict[weight_key].transpose(1, 0)]
                if bias_key in state_dict:
                    weights.append(state_dict[bias_key])
            elif layer_type == "BatchNormalization":
                weights = [
                    state_dict[weight_key],
                    state_dict[bias_key],
                    state_dict[f'{torch_layer_name}.running_mean'],
                    state_dict[f'{torch_layer_name}.running_var'],
                ]
            elif layer_type == "GroupNormalization" or layer_type == "LayerNormalization":
                weights = [
                    state_dict[weight_key],
                    state_dict[bias_key]
                ]
            elif layer_type == "Scale":
                weights = [
                    state_dict[f'{torch_layer_name}.scale']
                ]
            if len(weights) > 0:
                layer.set_weights(weights)

    def get_model(self):
        return self.model

    def _init_vars(self, input_shape=(640, 640, 3)):
        self.input_size = input_shape[:2]
        self.batched = True
        self.use_kps = False
        self._num_anchors = 1
        len_outputs = len(self.model.outputs)
        if len_outputs == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len_outputs == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len_outputs == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len_outputs == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        kr_blob = blob.transpose(0, 2, 3, 1)
        kr_net_outs = self.model(kr_blob)
        net_outs = [out.numpy() for out in kr_net_outs]

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size = None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ScrFD models in Keras')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='keras checkpoint file path')
    parser.add_argument('--input-imgs', type=str, help='images directory for input')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        # default=[640, 640],
        # default=[384, 384],
        default=[-1, -1],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.input_imgs:
        args.input_imgs = osp.join(osp.dirname(__file__), '../images')

    if len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0], 3)
    elif len(args.shape) == 2:
        input_shape = tuple(args.shape) + tuple([3])
    else:
        raise ValueError('invalid input shape')
    if input_shape[0] <= 0 or input_shape[1] <= 0:
        input_shape = (640, 640, 3)

    detector = SCRFD(args.config, input_shape)
    model = detector.get_model()
    model.load_weights(args.checkpoint)

    img_src_root = args.input_imgs
    assert os.path.isdir(img_src_root)
    img_dst_root = osp.join(osp.dirname(__file__), '../images_result')
    if not os.path.isdir(img_dst_root):
        os.makedirs(img_dst_root)

    img_paths = []
    for dirpath, _, filenames in os.walk(img_src_root):
        for file in filenames:
            path = os.path.join(dirpath, file)
            img_paths.append(path)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        print(f'input file: {img_path}')
        for _ in range(1):
            ta = datetime.now()
            bboxes, kpss = detector.detect(img, 0.5)
            tb = datetime.now()
            print('    all cost:', (tb-ta).total_seconds()*1000)
        print(f'    bboxes: {bboxes}')
        if kpss is not None:
            print(f'    kps shape: {kpss.shape}')
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(np.int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int)
                    cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
        filename = img_path.split('/')[-1]
        print('    output result:', filename)
        out_img_path = img_path.replace(img_src_root, img_dst_root)
        parent_dir = os.path.dirname(out_img_path)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
        cv2.imwrite(out_img_path, img)
