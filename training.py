import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import SegmentationNetwork, ClassificationNetwork, SubstitutionNetwork

from util.py.arg import getArgParser
from util.py.label import genOneHotLabelfromTDNandRCT
from util.tf.variables import initInputs, initPlaceHolders

args = getArgParser()

if __name__ == '__main__':

    """
    Parse Args
    """
    batch_size = args.batch
    input_w = args.w
    input_h = args.h
    input_channel = args.channel
    learning_rate = args.lr
    EPOCH = args.epoch
    weight_load_dir = args.weight_load_dir
    weight_save_dir = args.weight_save_dir
    save_interval = args.save_interval
    image_dir = args.image_dir

    input_shape = (input_h, input_w, input_channel)

    """
    Prepare dataset
    """
    mri_images = np.asarray([np.expand_dims(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=(input_w, input_h)), axis=-1) for img in glob.glob(f"{image_dir}/mri/*.png")])
    tdn_images = np.asarray([np.expand_dims(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=(input_w, input_h)), axis=-1) for img in glob.glob(f"{image_dir}/tendon/*.png")])
    rct_images = np.asarray([np.expand_dims(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=(input_w, input_h)), axis=-1) for img in glob.glob(f"{image_dir}/rct/*.png")])

    print(mri_images.shape)

    """
    Define Holder & Inputs
    """
    imgHolder, tdnHolder, rctHolder, segLabelHolder, clfLabelHolder = initPlaceHolders([
        np.zeros(shape=(batch_size, *input_shape), dtype=np.float32),
        np.zeros(shape=(batch_size, *(input_shape[:2]), 1), dtype=np.float32),
        np.zeros(shape=(batch_size, *(input_shape[:2]), 1), dtype=np.float32),
        np.zeros(shape=(batch_size, *(input_shape[:2]), 3), dtype=int),
        np.zeros(shape=(batch_size, 2), dtype=int)
    ])

    mri_input, tdn_input, rct_input = initInputs(input_shape=input_shape, num=3)

    """
    Define Models
    """
    # init clf & seg Models
    clfNet = ClassificationNetwork.Architecture().build_graph(input_shape=(None, *input_shape))
    segNet = SegmentationNetwork.Architecture().build_graph(input_shape=(None, *input_shape))

    # init substitution module
    sbtNet = SubstitutionNetwork.Architecture(vgg_shape=(input_shape[0], input_shape[1]))
    sbtOutput = sbtNet([mri_input, tdn_input, rct_input])
    sbtNet = keras.models.Model(inputs=[mri_input, tdn_input, rct_input], outputs=sbtOutput)

    # init clf-sbt module
    clf_sbt_Output = clfNet(sbtOutput['subst'])
    clfsbtNet = keras.models.Model(inputs=[mri_input, tdn_input, rct_input], outputs=clf_sbt_Output)


    @tf.function
    def define_losses():
        """
        Define loss functions
        """
        # L1-1 (classification): binary cross entropy loss
        L11 = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # [0, 1] norm
        loss11 = tf.reduce_mean(L11(clfLabelHolder, clfNet(imgHolder)))

        # L1-2 (substitution + classification): binary cross entropy loss
        L12 = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # [0, 1] norm
        loss12 = tf.reduce_mean(L12(clfLabelHolder, clfsbtNet([imgHolder, tdnHolder, rctHolder])))

        # L2 (segmentation): Categorical cross entropy loss
        L02 = tf.keras.losses.CategoricalCrossentropy()
        loss02 = tf.reduce_mean(L02(segLabelHolder, segNet(imgHolder)))

        # # L3 (substitution): contents & style loss
        model = sbtNet([imgHolder, tdnHolder, rctHolder])
        contentLoss = [tf.reduce_mean(model['tdn'][part]['content'][key] - model['rct'][part]['content'][key]) for part in ['real', 'imag'] for key in model['tdn']['imag']['content']]
        contentLoss = tf.add_n(contentLoss) / len(contentLoss)

        styleLoss = [tf.reduce_mean(model['tdn'][part]['style'][key] - model['rct'][part]['style'][key]) for part in ['real', 'imag'] for key in model['tdn']['imag']['style']]
        styleLoss = tf.add_n(styleLoss) / len(styleLoss)

        loss03 = tf.reduce_mean(contentLoss + styleLoss)

        # unite losses
        loss = tf.add_n([loss11 + loss12 + loss02 + loss03])
        return loss


    """
    Define Tensorflow Helpers
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    trainable_variables = clfNet.trainable_variables + segNet.trainable_variables + sbtNet.trainable_variables + clfsbtNet.trainable_variables
    eval = tf.keras.metrics.Mean()

    if weight_load_dir is not None:
        [model.load_weights(weight_load_dir) for model in [segNet, sbtNet, clfsbtNet]]

    """
    Training Process
    """
    for epoch in range(EPOCH):

        if epoch % save_interval == 0 and weight_save_dir is not None:
            [model.save_weights(weight_save_dir) for model in [clfNet, segNet, sbtNet, clfsbtNet]]

        """
        shuffling and augmentation can be added here
        """
        indexes = list(range(len(mri_images)))
        np.random.shuffle(indexes)
        cur = 0

        while cur < len(indexes):

            # mini-batch
            index = []
            while len(index) != batch_size:
                index.append(indexes[cur % len(indexes)])
                cur += 1
            index = np.asarray(index)

            # sampling
            mri = mri_images[index]
            tdn = tdn_images[index]
            rct = rct_images[index]

            seg_label = np.asarray(
                [
                    genOneHotLabelfromTDNandRCT(
                        tdn_mask=np.squeeze(tdn[i], axis=-1),
                        rct_mask=np.squeeze(rct[i], axis=-1)
                    ) for i in range(len(index))
                ]
            )

            clf_label = np.asarray(
                [
                    [1, 0] if np.mean(mask) > 0
                    else [0, 1]
                    for mask in rct
                ]
            )

            imgHolder.assign(mri)
            segLabelHolder.assign(seg_label)
            clfLabelHolder.assign(clf_label)

            with tf.GradientTape() as tape:
                loss = define_losses()

            # gradient descent
            grad = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grad, trainable_variables))
            eval(loss)

        print(f"({epoch + 1:03d}/{EPOCH:03d})", f"loss=[{eval.result():.5f}]")

        """
        Visualization
        """
        mask = segNet.predict(mri)
        cv2.imshow('img', (mask[0, :, :, 0] * (255. / 2.)).astype(np.uint8))
        cv2.waitKey(1)
