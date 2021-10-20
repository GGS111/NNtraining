import cv2
import torch
import numpy as np

import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

def resize(pimage):
    stream = open(pimage, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
    stream.close()

    bw_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    resize_image = cv2.resize(bw_, (128,128), interpolation = cv2.INTER_AREA)
    image_dims = np.expand_dims(resize_image/255, axis=2)
    return image_dims

def create_onnx(model,image,model_name):

    torch.onnx.export(model, image,
                    model_name,
                    export_params=True,
                    opset_version=10)

def load_onnx(onnx_model,image):
    session = onnxruntime.InferenceSession(onnx_model, None)
    input_name = session.get_inputs()[0].name
    image = np.expand_dims(resize(image),0)
    pred_onx = session.run(None, {input_name: image.astype(np.float32)})[0]
    print(pred_onx)