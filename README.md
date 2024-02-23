<h3>Description:</h3>
<p>As part of my End-of-Year Project (PFA) in artificial intelligence, with a focus on computer vision techniques, I have designed and implemented an intelligent embedded system capable of recognizing and sorting solid waste in real-time. This system utilizes deep neural networks for object classification. The libraries used for this project were Keras, TensorFlow, Numpy, Matplotlib, Open CV and Pandas. The embedded system is constructed around a Raspberry Pi 4 board equipped with a camera module for vision.
In this project, the objective is to train a Convolutional Neural Network (CNN) to classify an image into categories such as cardboard, glass, metal, paper, plastic, or trash.</p>
<h4>Schematic diagram of the sorting system:</h4>
<p>According to the specifications, our system is designed to sort waste received on a conveyor belt and classify them into 5 categories: plastic, glass, paper, steel, and others (organic, food...). As shown in Figure 1, this system includes a camera to capture images of the waste on the conveyor belt. These images will then be processed and analyzed in real-time using a Raspberry Pi 4 board embedded with image processing software based on deep neural networks (deep learning). 
This software classifies the processed images and controls scrapers to properly guide and transport the waste to the respective receiving bins (plastic, glass, paper, metal, cardboard, trash).</p>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/1b39ce25-42f3-4b1c-8b97-27e9fad32f66" alt="Your GIF" width="900" height="350">
<h3>Needs:</h3>
<p> -Jupyter Notebook</p>
<p> -Python</p>
<p> -Putty</p>
<p> -WinSCP</p>
<p> -VNC viewer</p>
<p> -Camera pi</p>
<p> -Raspberry Pi</p>
<h3>MODEL CONSTRUCTION</h3>
<p> We used a dataset of images downloaded from the internet via GitHub, which contains approximately 2400 images classified into 6 classes: cardboard, glass, paper, metal, trash, and plastic.
Here is the download link: https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip </p>
After downloading, the images were extracted. There are 5 steps in the deep learning process:
<h4>1-Load the data.</h4>
The pictures were taken by placing the object on a white posterboard and using sunlight and/or room lighting. The pictures have been resized down to 512 x 384,
<h4>2-Define the model.</h4>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/48cf6916-a12a-40d7-a98d-2006e0d65e9b" width="400" height="200">
<h4>3-Compile the model by calling the compile() function.</h4>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/28957869-0bdd-4e7f-ac9e-e9a49e3c4a05" width="400" height="200">
<h4>4-Train the model with the training dataset. Test the model on test data.</h4>
<p>Calling the compile() function</p>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/322d43d9-1a9e-4a25-b9f7-e3ada9e68064" width="400" height="200">
<h4>5-Make predictions by calling the evaluate() or predict() function.</h4>
<p>After defining the model and compiling it, we need to make predictions by running the model on the same data. Here we specify the epochs (epochs are the number of iterations for the training process to run in the dataset and the batch size is a number of instances that are evaluated before updating the weight)</p>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/d34af6b7-a8e1-4e52-91aa-3fd3d05d2a85" width="300" height="200">
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/cbc3b9bf-44f6-4aa0-bcfb-23b9c889f9b9" width="300" height="200">
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/9354fef2-d7e5-4816-a142-bca97f5fb200" width="300" height="200">
<h3>Installation</h3>
    
    source tflite-env/bin/activate 
    sudo apt -y install libjpeg-dev libtiff5-dev libjasper_dev libpng12-dev libavcoder-dev libavformat-dev libswscle-dev
    sudo apt -y install qt4-dev-tools libatlas-base-devlibhdf5-103
    
    python -m pip install of rib-python==4.1.0.25
<p>This is the ligne of command to run the code in raspberry pi</p>
<img src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/e4360a88-4ebf-4187-bd2a-bc9bdc9fb263" width="300" height="200">
<h3>Result</h3>
https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/d5683583-415c-46d3-b8d4-2b5f5712bd78
<video width="320" height="240" controls>
  <source src="https://github.com/yasminebs99/Detection-and-sorting-of-waste-using-Raspberry-pi/assets/160682389/d5683583-415c-46d3-b8d4-2b5f5712bd78" type="video/mp4">
</video>
