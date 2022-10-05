import glob

import cv2
import numpy as np

from model import SegmentationNetwork
from util.py.arg import getArgParserForInference
from util.tf.variables import initInputs


args = getArgParserForInference()

if __name__ == '__main__':

    """
    Parse Args
    """
    input_w = args.w
    input_h = args.h
    input_channel = args.channel
    weight_load_dir = args.weight_load_dir
    save_dir = args.output
    image_dir = args.source

    input_shape = (input_h, input_w, input_channel)

    """
    Prepare dataset
    """
    mri_images = np.asarray([np.expand_dims(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=(input_w, input_h)), axis=-1) for img in glob.glob(f"{image_dir}/*.png")])

    """
    Define Models
    """
    # init clf & seg Models
    segNet = SegmentationNetwork.Architecture().build_graph(input_shape=(None, *input_shape))

    if weight_load_dir is not None:
        [model.load_weights(weight_load_dir) for model in [segNet]]

    """
    Visualization
    """
    predictions = segNet.predict(mri_images)

    for index, prediction in enumerate(predictions):
        prediction = (prediction[:, :, 0] * (255. / 2.)).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/{index:03d}.png", prediction)
        cv2.imshow('prediction', prediction)
        cv2.waitKey(1000)
