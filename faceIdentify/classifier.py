# coding: utf-8
import os
import argparse

import cv2
import numpy as np

from core.face_extractor import FaceExtractor, show_image
from core.face_recognize import FaceClassifier

class Classifier:
    @classmethod
    def load_from_model(cls, filename, show):
        if not os.path.isfile(filename):
            print('Wrong model file path {}'.format(filename))
            exit()
        else:
            return cls(model=filename, show=show)

    @classmethod
    def train_from_folder(cls, root, save_model, show):
        return cls(folder=root, save_model=save_model, show=show)

    def __init__(self, show=False, model=None, folder=None, save_model=None):
        self.show = show
        self.frtool = FaceExtractor()
        self.fr_recog = FaceClassifier()

        if model is not None:
            self.fr_recog.load(model)

        elif folder is not None:
            self.fr_recog.train(folder, save_model)
        else:
            print('ERROR')
            exit()

    def predict_image(self, filename):
        if len(filename) == 0:
            return 
        elif not os.path.isfile(filename):
            print('Cannot found test file {}'.format(filename))
            return

        img = cv2.imread(filename)

        feature = self.frtool.extract_feature(img, self.show)
        if feature is None:
            print('Cannot found face in {}'.format(filename))
        else:
            result = self.fr_recog.recog_proba(feature)
            # dist = self.fr_recog.identify(feature)
            print(result)
            # dist = np.linalg.norm(self.features-feature, axis=1)
            # print(np.mean(dist))

    def predict_from_stream(self, web_cam_idx):
        if web_cam_idx == -1:
            return

        cam = cv2.VideoCapture(web_cam_idx)

        if not cam.isOpened:
            print('Cannot load cam stream')
        else:
            is_open = True
            ret, frame = cam.read()

            while ret and is_open:
                ret, frame = cam.read()
                feature = self.frtool.extract_feature(frame, show=False)

                flip_frame = cv2.flip(frame, 1)
                if feature is None:
                    is_open = show_image(flip_frame, text='No Face', is_block=False)
                else:
                    user, proba = self.fr_recog.recog_proba(feature)
                    # print(user, proba)
                    # avg_dist = self.fr_recog.recog(feature)
                    # print(dist)
                    # text = 'Similarity: {:.5f}'.format(avg_dist)
                    text = 'User: {}-{}'.format(user, np.max(proba))
                    is_open = show_image(flip_frame, text=text, is_block=False)

            
def create_argparser():
    parser = argparse.ArgumentParser(description='FaceRecognition')

    mutual = parser.add_mutually_exclusive_group(required=True)
    mutual.add_argument('--load', action='store_true', default=False, help='Load from model file (*.pkl)')
    mutual.add_argument('--train',action='store_true', default=False, help='Train model from subfolder of images in folder')

    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, help='From TRAIN, path to save model')
    parser.add_argument('--test', type=str, help='Test image', default='')
    parser.add_argument('--stream', type=int, help='Test from webcam', default=-1)

    parser.add_argument('--show', action='store_true', default=False, help='IS show frame')

    return parser.parse_args()

if __name__ == '__main__':
    args = create_argparser()
    print(args)

    if args.train:
        frtool = Classifier.train_from_folder(args.input, args.output, args.show)
    elif args.load:
        frtool = Classifier.load_from_model(args.input, args.show)

    frtool.predict_image(args.test)
    frtool.predict_from_stream(args.stream)