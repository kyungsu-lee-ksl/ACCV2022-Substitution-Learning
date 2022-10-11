## Preparations
### Download Codes
    $ git clone https://github.com/kyungsu-lee-ksl/ACCV2022-Substitution-Learning.git
    $ cd Template/
    $ sudo python -m pip install -r requirements.txt


### Pre-Trained Weights and Dataset

Due to intellectual property rights and privacy of medical data, please contact the author (ks_lee@dgist.ac.kr) for pre-trained weights, and datasets. You can find the sample images in ([[samples]](samples)).

<br>


### Primary Environment

The USG-Net was primarily implemented using Conda and TensorFlow 2 in the environment of Apple Silicon Chip. The dependency would be under _tensorflow-metal_ and _tensorflow-macos_.

    # xcode dependency
    $ xcode-select --install

    # opencv-python dependency
    $ conda install -c conda-forge opencv

    # Conda & Tensorflow dependency
    $ conda install -c apple tensorflow-deps==2.8.0

    # Additional Conda Library
    $ conda install pydot
    $ conda install graphviz

    # Tensorflow Dependency
    $ python -m pip install tensorflow-macos==2.8.0
    $ python -m pip install tensorflow-metal==0.4.0

<br>

### Training

    $ python training.py \
                --batch_size=<batch_size>
                --width=<width>
                --height=<height>
                --image_dir=<image_path>

See 'python training.py -h' for an overview of the system.


<br>

### Inference

    $ python inference.py \
                --weight_load=<weight_path> \
                --source=<images_path> \
                --output=<output_path>

<br>
<br>

## Substitution Learning

### A. Network Architecture
[[Code]](model)

<div>
<p align="center">
    <img src="assets/dense_block.png" width="100%"\>
</p>
Fig. 1. Dense Block in the CSS-Net. Since ST and RCT occupy a small area, it is required to enlarge receptive fields to comprehend the correlations between RCT and ST. To this end, the Astros convolutions are utilized.
</div><br /><br />

<div>
<p align="center">
    <img src="assets/model.png" width="100%"\>
</p>
Fig. 2. (a) Overall pipeline of CSS-Net. Each module is optimized using the corresponding loss functions in the training phase. In the inference phase, the CSS-Net predicts the segmentation masks of ST and RCT using only the segmentation module. (b) Detailed architecture of the substitution learning module.
</div><br /><br />

Fig. 2(a) describes the overall architecture and pipeline of the CSS-Net, which includes the substitution $M_{sl}$, classification $M_{clf}$, and segmentation $M_{seg}$ modules based on convolutional neural networks (CNNs). The CSS-Net aims to predict the multi-categorical segmentation masks of the background (BG), ST, and RCT. To this end, the CSS-Net is mainly designed for the segmentation task (main task). In addition, despite the feasibility of single utilization of $M_{seg}$, to enhance the feature extraction during the optimization, two supplementary tasks and modules are appended; the classification module $M_{clf}$ and the substitution learning module $M_{sl}$. Figs. 1 and 2 illustrate the detailed architecture of the CSS-Net. Note that, several convolutions are shared between $M_{seg}$ and $M_{clf}$ to transfer the learned knowledge related to RCT as illustrated in Fig. 2.

##

### B. Substitution Learning
[[Code]](model/SubstitutionNetwork.py)

$M_{sl}$ substitutes the RCT region, which is a ruptured ST area in the MRI images, for a normal ST style. The substituted images are then utilized as additional inputs for $M_{seg}$ and $M_{clf}$, in terms of data augmentation. First, $I$ is binary-masked using the corresponding ground truth $G=G^{seg}$ with the two outputs $I \ast G(c=1)$ and RCT $I \ast G(c=2)$, where $\ast$ indicates the Hadamard product. The individual masked regions are then converted into the frequency-domain as shown in Figs. Here, the DFT is formulated as follows:

```math
  F[x, y] = \frac{1}{HW}\sum_{h}^{H}\sum_{w}^{W} I[h, w] e^{-j2\pi \big(\frac{h}{H}y + \frac{w}{W}x\big)}, \;\; j=\sqrt{-1}
```

where, $F$ is the output mapped into the frequency domain, $e$ is Euler's number, and $H$ and $W$ are the height and width of $I$, respectively. Subsequently, a simple CNN architecture $S$ with identical mapping transfers the DFT-converted output of the RCT, which is $D=\textit{DFT}(I \ast G(c=2))$, as $D'=(S \circ \textit{DFT})(I \ast G(c=2))$. The inverse DFT (IDFT) is applied to $D'$, and the substituted images $\textit{IDFT}(D')$ is finally generated. In summary, the substituted image $I'$ is calculated as $I' = M_{sl}(I) =(\textit{IDFT} \circ S \circ \textit{DFT})(I \ast G(c=2))$. The generated images by SL are more reliable than those of GANs.


##


<br>
<br>

## [PAPER] CSS-Net: Classification and Substitution for Segmentation of Rotator Cuff Tear

#### Authors
Kyungsu Lee, Hah Min Lew, Moonhwan Lee, Jun-Young Kim, and Jae Youn Hwang*

#### Abstract
Magnetic resonance imaging (MRI) has been popularly used to diagnose orthopedic injuries because it offers high spatial resolution in a non-invasive manner. Since the rotator cuff tear (RCT) is a tear of the supraspinatus tendon (ST), a precise comprehension of both is required to diagnose the tear. However, previous deep learning studies have been insufficient in comprehending the correlations between the ST and RCT effectively and accurately. Therefore, in this paper, we propose a new method, \textit{substitution learning}, wherein an MRI image is used to improve RCT diagnosis based on the knowledge transfer. The \textit{substitution learning} mainly aims at segmenting RCT from MRI images by using the transferred knowledge while learning the correlations between RCT and ST. In substitution learning, the knowledge of correlations between RCT and ST is acquired by substituting the segmentation target (RCT) with the other target (ST), which has similar properties. To this end, we designed a novel deep learning model based on multi-task learning, which incorporates the newly developed substitution learning, with three parallel pipelines: (1) segmentation of RCT and ST regions, (2) classification of the existence of RCT, and (3) substitution of the ruptured ST regions, which are RCTs, with the recovered ST regions. We validated our developed model through experiments using 889 multi-categorical MRI images. The results exhibit that the proposed deep learning model outperforms other segmentation models to diagnose RCT with $6 \sim 8$\% improved IoU values. Remarkably, the ablation study explicates that substitution learning ensured more valid knowledge transfer.


#### Experimental Results

<p align="center">
    <img src="assets/exp1.png" width="100%"\>
</p>

Table 5 illustrates the quantitative analysis of the proposed CSS-Net compared with other deep learning models. U-Net and DeepLabV3+ were employed because of their popularity in segmentation tasks. In addition, the SA-RE-DAE and IteR-MRL were utilized as state-of-the-art segmentation and multi-task models. The experimental results demonstrated that all models achieve high scores in segmenting BG. Expecting the RCT, the CSS-Net significantly outperforms the other models. It showed at least a 6\% IoU-RCT compared to the other models. In particular, the CSS-Net achieved 10\% $\sim$ 20\% improved sensitivity in RCT segmentation, suggesting that the CSS-Net with substitution learning could be utilized as an excellent diagnostic tool to localize RCT, as shown in Fig. 6. 


<p align="center">
    <img src="assets/exp2.png" width="100%"\>
</p>

Representative results of Guided Grad-CAMs. Left$\rightarrow$Right: MRI images $I$, Ground truth $G$, Guided-backprop $B$, Overlay of $I$ and Grad-CAM $A_{\textit{Clf}}$ by the \textit{Clf}-network which has only classification module, Overlay of $I$ and Grad-CAM $A_{\textit{Clf+SL}}$ by the \textit{Clf+SL}-network which has classification and substitution module, Guided Grad-CAM $B \ast A_{\textit{Clf}}$ by the \textit{Clf}-network, and Guided Grad-CAM $B \ast A_{\textit{Clf+SL}}$ by the \textit{Clf+SL}-network.


#### Conclusion

We introduced integrated multi-task learning as an end-to-end network architecture for RCT segmentation in MRI images. We also proposed a novel substitution learning using DFT to augment data more reliably for the imbalanced dataset, as well as to improve accuracy by knowledge transfer. We employed the SL instead of GANs-based approaches since the SL was demonstrated as more reliable than GANs with even low computation costs. Our results showed that the CSS-Net produced a superior segmentation performance owing to the abundant knowledge transfer from the classification and substitution tasks to the segmentation task, outperforming other state-of-the-art models. It showed a 10\% higher IoU value than the baseline model, and even at least 6\% higher IoU values than those shown by other state-of-the-art models. Further experiments should be conducted for clinical applications that require reliable data augmentation and high performance.


## Citation

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  pages={},
  year={2022},
  organization={}
}
```
