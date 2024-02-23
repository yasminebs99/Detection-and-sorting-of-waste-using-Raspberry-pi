from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time
import tensorflow as tf
import cv2
def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, frame):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = frame

def classify_image(interpreter, frame):
  set_input_tensor(interpreter, frame)
  top_k=1
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  print(interpreter.get_output_details()[0])
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

 ### scale, zero_point = output_details['quantization']
  ###output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

data_folder = "/home/pi/Projects/Python/tflite/tri/"

model_path = data_folder + "f_lite_model.tflite"
label_path = data_folder +"model/label.pbtxt"

interpreter = tf.lite.Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

# Load an image to be classified.
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
     
#IM_WIDTH = 1280
#IM_HEIGHT = 720
#IM_WIDTH = 640   # Use smaller resolution for
#IM_HEIGHT = 480

    # Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.start_preview()
camera.resolution = (224, 224)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(224, 224))
camera.start_recording('/home/pi/Desktop/video122.h264')
time.sleep(0.1)

    # Continuously capture frames and perform object detection on them
for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    
    
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame =np.asarray(frame1.array)
    rawCapture.truncate(0)
    time1 = time.time()
    label_id, prob = classify_image(interpreter, frame)
  
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    # Read class labels.
    labels = load_labels(label_path)

    # Return the classification label of the image.
    classification_label = labels[label_id]
    print("Image Label is :", classification_label, ", with Accuracy :", prob*100, "%.")    
    
    
camera.stop_recording()
camera.stop_preview()

####image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))

# Classify the image.
