FaceIdentify Application
---
* Train from folder and save model
* Load model and test from image or webcam

## Feature
1. Accelerate speed by resize input image.
2. Light resistance

## Download model first
- [Shape Predictor Model](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2): put into `$PROJECT/resources/`
- [Face Recog Model](https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2): put into `$PROJECT/resourcces/`

## Usage:
### Train from folder
- `$ python performance.py --train --input <IMAGE FOLDER> --output <MODEL DUMP(*.npy)>`

### Load from model file(*.npy)
- `$ python performance.py --load --input <MODEL(*.npy)>`

### Test image similarity
- `$ python performance.py --load --input <MODEL(*.npy> --test <IMAGE PATH>`

### Test from webcam
- `$ python performance.py --load --input <MODEL(*.npy> --stream <WEBCAM INDEX>`

### Additional
Argument::`show` will display image with result. If this argument enable in training, it will block progress until user click.


