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

## Identify(one person)
### Train from folder
- `$ python identify.py --train --input <IMAGE FOLDER> --output <MODEL DUMP(*.npy)>`

### Load from model file(*.npy)
- `$ python identify.py --load --input <MODEL(*.npy)>`

### Test image similarity
- `$ python identify.py --load --input <MODEL(*.npy> --test <IMAGE PATH>`

### Test from webcam
- `$ python identify.py --load --input <MODEL(*.npy> --stream <WEBCAM INDEX>`

### Additional
Argument::`show` will display image with result. If this argument enable in training, it will block progress until user click.


## Classifier(multiple person)
### Train from folder
> This folder should contain multiple folder to indicate different person.
- `$ python classifier.py --train --input <IMAGE FOLDER> --output <MODEL DUMP(*.pkl)>`

### Load from model file(*.pkl)
- `$ python classifier.py --load --input <MODEL(*.pkl)>`

### Test image similarity
- `$ python classifier.py --load --input <MODEL(*.pkl)> --test <IMAGE PATH>`

### Test from webcam
- `$ python classifier.py --load --input <MODEL(*.pkl)> --stream <WEBCAM INDEX>`
