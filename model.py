"""
Script to configure model or 
use a pre-trained models. 

"""


from keras.optimizers import SGD
import convnets





####################################################
# IF USING PRE-TRAINED MODELS
# OPRIONS - 1. Alexnet
#           2. VGG 16 layers
####################################################

def get_trained_model():
    """
    Function return Alexnet model trained on 
    one million images of Imagenet dataset. 

    """
    model = convnets.convnet('alexnet',weights_path= "/scratch/sk1846_data/trainedModels/alexnet_weights.h5",heatmap=False)

    return model 
    


