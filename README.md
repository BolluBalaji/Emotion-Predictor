# Emotion-Predictor

Detecting Emotions Using deep CNN Architecture<br/><br/>
**Steps for Running**<br/>
1 .First download and install all necessary libraries <br/>
   - [Python](https://www.python.org) 3.6.8 or any latest version <br/>
   - [opencv](https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html)<br/>
   - [Keras](https://pypi.org/project/Keras/) <br/>
   - [Tensorflow](https://www.tensorflow.org/install) <br/>
   - [dlib](https://pypi.org/project/dlib/) 
2. Next download the dataset from Kaggle<br/>
   - [DataSet](https://www.kaggle.com/c/facial-keypoints-detector/overview)<br/><br/>
3. Training <br/>
   - If you want to train the model from scratch use Colab [Notebook](https://github.com/BolluBalaji/Emotion-Predictor/blob/master/CNN_Model.ipynb)<br/>
   - or Use [Pretrained](https://github.com/BolluBalaji/Emotion-Predictor/blob/master/Emotion_1.h5) Model<br/><br/>
 4. For Live Detection <br/>
   * i) [HaarCascade](https://github.com/BolluBalaji/Emotion-Predictor/blob/master/haarcascade_frontalface_default.xml) Face Detector<br/>
      - Run [HaarCascade_Face_Recog.py](https://github.com/BolluBalaji/Emotion-Predictor/blob/master/HaarCascade_Face_Recog.py)<br/>
  * ii)[HOG](https://pypi.org/project/hog/) Histogram of oriented gradients Face Detector<br/>
      - Run [Hog_Face_Recog.py](https://github.com/BolluBalaji/Emotion-Predictor/blob/master/Hog_Face_Recog.py)<br/>

