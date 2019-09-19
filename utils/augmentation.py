import numpy as np

from keras.applications.xception import preprocess_input

import imgaug as ia
from imgaug import augmenters as iaa

def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 50% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.6), "y": (0.9, 1.6)}, #>20 will cut part of img
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # >20% will also cut part of img
                rotate=(-10, 10), # 45/-45° -> works good with scale + translate to prevent cuts
                shear=(-5, 5), # shear by -16 to +16 degrees
                mode=ia.ALL 
            )),
            iaa.SomeOf((0, 4), [
                    sometimes(iaa.Superpixels(p_replace=(0.3, 0.7), n_segments=(10, 100))), #superpixel-representation --> better basallamina representation 
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.2)), #small blur effects --> better representation
                        iaa.AverageBlur(k=(1, 3)), # k must be odd
                        iaa.MedianBlur(k=(1, 3)), # 
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), #cell wall represenation
                    iaa.Emboss(alpha=(0, 0.8), strength=(0, 0.5)), #cell wall represenation
                    #searching for edges or angles --> blobby mask --> better basallamina representation / nuclei
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.2, 0.4)), #detects edges --> cell wall,..
                        iaa.DirectedEdgeDetect(alpha=(0.2, 0.4), direction=(0.0, 1.0)), #direction will make edges from random directions 
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.2), # add gaussian noise to images
                 iaa.OneOf([
                        iaa.Dropout((0.05, 0.3), per_channel=0.2), #rnd remove 5-30% in small pixels
                        iaa.CoarseDropout((0.05, 0.3), size_percent=(0.01, 0.02), per_channel=0.2),# rnd remove 3% in big pixels
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.3), # change brightness of images (by -10 to 10 of original value)
                    #iaa.AddToHueAndSaturation((-0.1, 0.1)), # change hue and saturation
                    #
                    #either change the brightness of the whole image (sometimes per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.2), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.1)), #still not sure: move pixels locally around
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))), #still not sure:move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                         random_order=True
            )
        ],
        random_order=True
    )
    return seq

def aug(X):
    seq = get_seq()
    X=[X]  
    X=seq.augment_images(X) 
    X=np.asarray(X[0])
    X=preprocess_input(X.astype(np.float32))
    return X           