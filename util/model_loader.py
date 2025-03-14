"""Keras model loader"""

import os
from keras.models import load_model
import torch


class ModelLoader:
    """Keras model loader"""

    @staticmethod
    def loadfromfile(modelpath):
        """Load model from filepath"""
        # print(torch.__version__)
        assert os.path.exists(modelpath) == 1

        if modelpath.endswith(".h5"):
            model = load_model(modelpath)
        elif (
            modelpath.endswith(".pt")
            or modelpath.endswith(".pth")
            or modelpath.endswith(".ckpt")
        ):
            model = torch.load(modelpath, map_location="cpu")
        # elif modelpath.endswith('.tar'):
        #    tarmodel =  torch.load(modelpath, map_location='cpu')
        #    model = tarmodel['state_dict']

        return model
