# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import division

from datetime import datetime
import os
import os.path as osp
import time
import argparse

import cv2
import numpy as np
import tensorflow as tf


def load_tflite_model(path, debug=False):
    print('##################################\nLoad {} model'.format(path))
    # Load TFLite model and allocate tensors.
    interpreter_ = tf.lite.Interpreter(model_path=path)
    interpreter_.allocate_tensors()

    # Get input and output tensors.
    input_details_ = interpreter_.get_input_details()
    if debug:
        print('{} input(s): {}'.format(len(input_details_), input_details_))
    output_details_ = interpreter_.get_output_details()
    if debug:
        print('{} output(s): {}'.format(len(output_details_), output_details_))

    return interpreter_, input_details_, output_details_


def inference_calculator(interpreter_, input_details_, output_details_, tensor_image, outputs_order=None, debug=False):
    for _id, _input in enumerate(input_details_):
        interpreter_.set_tensor(_input['index'], tensor_image[_id])

    start_invoke_time = time.time()
    interpreter_.invoke()
    if debug:
        print('invoke time: {} ms'.format(int(1000 * (time.time() - start_invoke_time))))

    results_list = []
    if outputs_order is None:
        for _output in output_details_:
            results_list.append(interpreter_.get_tensor(_output['index']))
    else:
        for _output_index in outputs_order:
            results_list.append(interpreter_.get_tensor(output_details_[_output_index]['index']))
    return results_list


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
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
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, interpreter=None):
        self.model_file = model_file
        self.interpreter = interpreter
        self.taskname = 'detection'
        self.batched = False
        if self.interpreter is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.interpreter, self.input_details, self.output_details = load_tflite_model(self.model_file)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        # print('input(s):')
        # for node in self.input_details:
        #     print('    name:', node['name'], ', shape:', node['shape'])
        input_cfg = self.input_details[0]
        input_shape = input_cfg['shape']
        if isinstance(input_shape[1], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[1:3][::-1])
        input_name = input_cfg['name']
        # print('output(s):')
        # for node in self.output_details:
        #     print('    name:', node['name'], ', shape:', node['shape'])
        outputs = self.output_details
        if len(outputs[0]['shape']) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o['name'])
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
        # set outputs order
        self.outputs_order = []
        for _index in range(len(outputs)):
            for _idx, output in enumerate(outputs):
                _order = int(output["name"].split(":")[1])
                if _order == _index:
                    self.outputs_order.append(_idx)

    def prepare(self, ctx_id, **kwargs):
        # if ctx_id < 0:
        #     self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        tflite_blob = blob.transpose(0, 2, 3, 1)
        net_outs = inference_calculator(self.interpreter, self.input_details, self.output_details, [tflite_blob],
                                        self.outputs_order)
        # print('net_outs')
        # for net_out in net_outs:
        #     print('    ', net_out.shape)

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
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size=None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
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
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
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
            if metric == 'max':
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
        description='Run ScrFD models in TFLite')
    parser.add_argument('model_file', help='tflite model file path')
    parser.add_argument('--input-imgs', type=str, help='images directory for input')
    args = parser.parse_args()
    return args


def parse_shape(model_file):
    from pathlib import Path
    model_name = Path(model_file).stem
    if "shape" not in model_name:
        return (640, 640)
    shape = model_file.split("shape")[1].split("x")
    return shape


if __name__ == "__main__":
    args = parse_args()

    if not args.input_imgs:
        args.input_imgs = osp.join(osp.dirname(__file__), '../images')

    input_shape = parse_shape(args.model_file)
    detector = SCRFD(model_file=args.model_file)
    detector.prepare(-1)

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
