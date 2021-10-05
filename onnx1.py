import cv2
import torch
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare

import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx

def resize_(pimage):
    stream = open(pimage, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
    stream.close()

    bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
    resize_image = cv2.resize(bw_, (128,128), interpolation = cv2.INTER_AREA)

    return resize_image/255

def create_onnx(model,image,model_name):

    model.eval()

    torch.onnx.export(model, image,
                    model_name,
                    export_params=True,
                    opset_version=10)

def load_onnx(onnx_model,image):
    session = onnxruntime.InferenceSession(onnx_model, None)
    input_name = session.get_inputs()[0].name
    image = np.expand_dims(resize_(image),0)
    pred_onx = session.run(None, {input_name: image.astype(np.float32)})[0]
    return pred_onx

def load_model(onnx_model):
    return onnx.load(onnx_model)

model = load_model("C:/python/NNtraining/VGG.onnx")

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)
print('tf_rep',tf_rep)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

tf_rep.export_graph("C:/python/NNtraining/VGG.pb")

converter =  tf.compat.v1.lite.TFLiteConverter.from_saved_model("C:/python/NNtraining/VGG.pb")
# tflite_model = converter.convert()
# open("C:/python/NNtraining/mnist.pb", "wb").write(tflite_model)

