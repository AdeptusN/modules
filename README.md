# modules
Repository of NN modules, metrics, losses and transforms by AdeptusN

## conv_modules.py

Contains custom convolutional neural net modules:

- Conv3x3 - convolution module with optional batch_norm, dropout layers and activation function;
- Conv5x5 - convolution module that replaces conv layer with kernel_size=5 by 2 conv layers with kernel_size=3.
    Have optional Dropout and BatchNorm layers after every conv layer and optional residual connection;
- Conv7x7 - convolution module that replaces conv layer with kernel_size=7 by 3 conv layers with kernel_size=3.
    Have optional Dropout and BatchNorm layers after every conv layer and optional residual connection;
    
## losses.py

Contains custom losses:

- FocalLossBin - focal loss for binary classification;
- FocalLossMulti - [IN TEST] focal loss for multiclass classification [IN TEST];
- L1Loss - based on manhattan distance loss;
- VAELoss - a loss for variational autoencoder;
- PerceptualLoss - perceptual loss calculate MSE loss between vgg16 activation maps
    of model output image and target image;
- WassersteinLoss - a loss for discriminator of generative adversarial network;
- GradientPenalty - gradient penalty (for Wasserstein loss);

## metrics.py

Contains custom metrics:

- IoUMetricBin - intersection over union metric for binary segmentation;
- DiceScoreBin - metric for binary segmentation based on dice-coefficient;

## transforms.py

Contains custom transforms:

- MinMaxScale - Min-Max scale to transform image pixels to [0, 1];
- ThresholdTransform - threshold image transformation;
- DividerScaler - scalar division of image tensor (can be added to torchvision.transforms.Compose);
