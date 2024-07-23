import argparse
import os
from os import path as osp

import torch

from mmdet_keras.scrfd import SCRFD


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export ScrFD model to Keras')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--to-tflite', action="store_true", default=False, help='Convert to TFLite')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[-1, -1],
        help='tflite model input shape')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

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

    # Convert torch weights to keras weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    detector.update_weights({k: state_dict[k].numpy() for k in state_dict})

    if len(args.output_file) == 0:
        output_dir = osp.join(osp.dirname(__file__), '../models')
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        cfg_name = args.config.split('/')[-1]
        pos = cfg_name.rfind('.')
        cfg_name = cfg_name[:pos]
        args.output_file = osp.join(output_dir, "%s.h5" % cfg_name)
        if args.to_tflite:
            tflite_output_file = osp.join(output_dir, "%s_shape%dx%d.tflite" % (cfg_name, input_shape[0], input_shape[1]))

    # Export to h5
    model.save_weights(args.output_file, overwrite=True)
    print("Keras model exported successfully!")

    # Export to tflite
    if args.to_tflite:
        import tensorflow as tf
        run_model = tf.function(lambda x: model(x))
        batch_size = 1
        input_tensor_shape = tuple([batch_size]) + input_shape
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(input_tensor_shape, model.inputs[0].dtype))

        saved_model_dir = args.output_file.replace(".h5", "")
        model.save(saved_model_dir, save_format="tf", signatures=concrete_func)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        with open(tflite_output_file, "wb") as fo:
            fo.write(tflite_model)

        import shutil
        shutil.rmtree(saved_model_dir)
        print("TFLite model exported successfully!")

    # print("DEBUGGING...")

    # model.summary()
    # layers_count = 0
    # for l_id, layer in enumerate(model.layers[1:]):
    #     print(f"Layer {l_id}: {type(layer).__name__}\n")
    #     # print(f"    Config: {layer.get_config()}")
    #     for w_id, weight in enumerate(layer.get_weights()):
    #         print(f"    w{w_id}: {weight.shape}")
    #     layers_count += 1
    # print(f"Total number of layers: {layers_count}\n")
    # from keras.utils.layer_utils import count_params
    # tf_trainable_params = count_params(model.trainable_weights)
    # print(f"Total number of trainable parameters: {tf_trainable_params}\n")

    # input_npy = osp.join(osp.dirname(__file__), '../npy/tensor_data.npy')
    # tensor_data = np.load(input_npy).transpose(0, 2, 3, 1)
    # keras_result = model(tensor_data)

    # test final output
    # for i, out in enumerate(keras_result):
    #     print(f"output {i}, out.shape = {out.shape}")
    #     npy_out = np.load('./npy/tensor_result_{}.npy'.format(i))
    #     diff = out - npy_out
    #     print("output n0. {}, L2 distance = {}".format(i, np.linalg.norm(diff)))
    #     print("output n0. {}, mean difference = {}".format(i, np.mean(np.abs(diff))))

    # test head
    # for i, out in enumerate(keras_result):
    #     pt_out = np.transpose(out, (0, 3, 1, 2))
    #     print(pt_out[0, 1, 10, 10])
    #     npy_pt_out = np.load('./npy/head_{}.npy'.format(i))
    #     print(npy_pt_out[0, 1, 10, 10])
    #     diff = pt_out - npy_pt_out
    #     print("output n0. {}, L2 distance = {}".format(i, np.linalg.norm(diff)))
    #     print("output n0. {}, mean difference = {}".format(i, np.mean(np.abs(diff))))

    # test neck
    # for i, out in enumerate(keras_result):
    #     pt_out = np.transpose(out, (0, 3, 1, 2))
    #     npy_pt_out = np.load('./npy/neck_{}.npy'.format(i))
    #     diff = pt_out - npy_pt_out
    #     print("output n0. {}, L2 distance = {}".format(i, np.linalg.norm(diff)))
    #     print("output n0. {}, mean difference = {}".format(i, np.mean(np.abs(diff))))

    # test backbone
    # for i, out in enumerate(keras_result):
    #     pt_out = np.transpose(out, (0, 3, 1, 2))
    #     npy_pt_out = np.load('./npy/backbone_{}.npy'.format(i))
    #     diff = pt_out - npy_pt_out
    #     print("output n0. {}, L2 distance = {}".format(i, np.linalg.norm(diff)))
    #     print("output n0. {}, mean difference = {}".format(i, np.mean(np.abs(diff))))
