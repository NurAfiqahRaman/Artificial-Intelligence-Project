# HUMAN FACIAL EMOTION RECOGNITION

## A. PROJECT SUMMARY

**Project Title:** Human Facial Emotion Detection

**Team Members:** 
- Tan Wei Yin
- Pang Jia Mei
- Nur Afiqah Binti Raman
- Mirza Sahid Afridi

- [ ] **Objectives:**
- in entertainment industries: to propose the most appropriate entertainment for the target audience
- in robotics: to design smart collaborative or service robots which can interact with humans
- in marketing: to create specialized adverts, based on the emotional state of the potential customer


##  B. ABSTRACT 

Automated emotion recognition (AEE) is an important issue in various fields of activities which use human emotional reactions as a signal for marketing, technical equipment, or human–robot interaction. The classification of emotion sensors is presented to reveal area of application and expected outcomes from each method, as well as their limitations.


![Coding](https://github.com/NurAfiqahRaman/Artificial-Intelligence-Project/blob/main/facial%20emotion.png)
Figure 1 shows the AI output of detecting the emotion of the user from their facial expression.


## C.  DATASET

In this project, we’ll discuss our human facial emotion detector, detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our custom human emotion detector.

I’ll then show you how to implement a Python script to train a human emotion detector on our dataset using Keras and TensorFlow.

We’ll use this Python script to train a human emotion detector and review the results.

Given the trained facial emotion detector, we’ll proceed to implement two more additional Python scripts used to:

- Detect human emotion in images
- Detect human emotion in real-time video streams

We’ll wrap up the post by looking at the results of applying our human emotion detector.


There is two-phase COVID-19 face mask detector as shown in Figure 2:

![Figure 2](https://github.com/NurAfiqahRaman/Artificial-Intelligence-Project/blob/main/dataset%20phase.jpg)
Figure 2: Phases and individual steps for building a COVID-19 face mask detector with computer vision and deep learning 

In order to train a custom human emotion detector, we need to break our project into two distinct phases, each with its own respective sub-steps (as shown by Figure 1 above):

- Training: Here we’ll focus on loading our human emotion detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the human emotion detector to disk

- Deployment: Once the human emotion detector is trained, we can then move on to loading the emotion detector, performing face detection, and then classifying each face as sad or angry or happy 

We’ll review each of these phases and associated subsets in detail in the remainder of this tutorial, but in the meantime, let’s take a look at the dataset we’ll be using to train our human emotion detector.


Our human emotion detection dataset as shown in Figure 3:

![Figure 3](https://github.com/NurAfiqahRaman/Artificial-Intelligence-Project/blob/main/emotion%20dataset.png)

Figure 3: A human emotion detection dataset consists of “sad” , “angry” and “happy” images. 

The dataset we’ll be using here today was created by Kagle reader Adam.

This dataset consists of 1,376 images belonging to two classes:

- sad: 1247 images
- angry: 958 images
- happy : 1774 images
- 
Our goal is to train a custom deep learning model to detect whether a person is sad or angry or happy.

How was our human emotion dataset created?
Adam, a restaurant owner need his customer's feedback about his restaurant and food to improve his restaurant quality. Some customers are not loyal with what their wrote and said when he asked about the feedback as the customer do not want him to feel down.

To help keep his upgrade his business, Adam decided to distract himself by applying computer vision and deep learning to solve a real-world problem:

- Best case scenario — he could use her project to help others
- Worst case scenario — it gave him more wondering on how to maintain good quality


## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --filelimit 10
- .
- ├── dataset
- │   ├── sad [1247 entries]
- │   └── angry [958 entries]
- │   └── happy [1774 entries]
- ├── examples
- │   ├── example_01.png
- │   ├── example_02.png
- │   └── example_03.png
- ├── emotion_detector
- │   ├── deploy.prototxt
- │   └── res10_300x300_ssd_iter_140000.caffemodel
- ├── detect_emotion_image.py
- ├── detect_emotion_video.py
- ├── emotion_detector.model
- ├── plot.png
- └── train_emotion_detector.py
- 5 directories, 10 files


The dataset/ directory contains the data described in the “Our human emotion detection dataset” section.

Three image examples/ are provided so that you can test the static image face mask detector.

We’ll be reviewing three Python scripts in this tutorial:

- train_emotion_detector.py: Accepts our input dataset and fine-tunes MobileNetV2 upon it to create our emotion_detector.model. A training history plot.png containing accuracy/loss curves is also produced
- detect_emotion_image.py: Performs human emotion detection in static images
- detect_emotion_video.py: Using your webcam, this script applies human emotion detection to every frame in the stream

In the next two sections, we will train our face mask detector.



## E   TRAINING THE COVID-19 FACE MASK DETECTION

We are now ready to train our human emotion detector using Keras, TensorFlow, and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python train_emotion_detector.py --dataset dataset
- [INFO] loading images...
- [INFO] compiling model...
- [INFO] training head...
- Train for 34 steps, validate on 276 samples
- Epoch 1/20
- 34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
- Epoch 2/20
- 34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
- Epoch 3/20
- 34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
- Epoch 4/20
- 34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
- Epoch 5/20
- 34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
- ...
- Epoch 16/20
- 34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
- Epoch 17/20
- 34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
- Epoch 18/20
- 34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
- Epoch 19/20
- 34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
- Epoch 20/20
- 34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922
- [INFO] evaluating network...

|      |    precision    | recall| f1-score | support |
|------|-----------------|-------|----------|---------|
|sad|0.99|1.00|0.99|138|
|angry|1.00|0.99|0.99|138|
|happy|1.00|0.99|0.99|138|
|accuracy| | |0.99|276|
|macro avg|0.99|0.99|0.99|276|
|weighted avg|0.99|0.99|0.99|276|


![Figure 4](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detector_plot.png)

Figure 4: Figure 10: human emotion detector training accuracy/loss curves demonstrate high accuracy and little signs of overfitting on the data

As you can see, we are obtaining ~99% accuracy on our test set.

Looking at Figure 4, we can see there are little signs of overfitting, with the validation loss lower than the training loss. 

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting human emotion with OpenCV in real-time

You can then launch the human emotion detector in real-time video streams using the following command:
- $ detect_emotion_video.py
- [INFO] loading human emotion detector model...
- INFO] loading human emotion detector model...
- [INFO] starting video stream...

[![Figure5](https://img.youtube.com/vi/XVQSMbeBGZQ/0.jpg)](https://www.youtube.com/watch?v=XVQSMbeBGZQ "Figure5")

Figure 5: Human emotion in real-time video streams

In Figure 5, you can see that our human emotion detector is capable of running in real-time (and is correct in its predictions as well.



## G.   PROJECT PRESENTATION 

In this project, you learned how to create a human facial emotion detector using OpenCV, Keras/TensorFlow, and Deep Learning.

To create our human emotion detector, we trained a 3-class model of people showing sad, angry and happy face reaction .

We fine-tuned MobileNetV2 on our mask/no mask dataset and obtained a classifier that is ~99% accurate.

We then took this human emotion classifier and applied it to both images and real-time video streams by:

- Detecting faces in images/video
- Extracting each individual face
- Applying our human emotion classifier

Our human emotion detector is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

[![demo](https://img.youtube.com/vi/-AP9e4ny_KHc/0.jpg)](https://www.youtube.com/watch?v=AP9e4ny_KHc "demo")




