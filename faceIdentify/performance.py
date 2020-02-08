# coding: utf-8
import enum
import pickle
import os
import argparse

import numpy as np
import cv2
import dlib

@enum.unique
class FaceDirection(enum.IntEnum):
    FRONT = 0
    LEFT = 1
    RIGHT = 2
    FRONT_ROTATE_LEFT = 3
    FRONT_ROTATE_RIGHT = 4

def _rect_to_bbox(rect):
    return rect.left(), rect.top(), rect.right(), rect.bottom()

def show_image(img, bbox=None, text=None, is_block=True):
    if bbox is not None:
        img = cv2.rectangle(img, 
            (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 5)

    if text is not None and len(text):
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow('image', img)
    if is_block:
        cv2.waitKey(0)
    else:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):                                                                                                               
           return False 
    return True

class FaceExtractor:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

        shape_predictor_filename = 'resources/shape_predictor_68_face_landmarks.dat'
        face_recognition_model_filename = 'resources/dlib_face_recognition_resnet_model_v1.dat'

        if not os.path.exists(shape_predictor_filename):
            print('Please put shape_prector model in path::{}'.format(shape_predictor_filename))
        if not os.path.exists(face_recognition_model_filename):
            print('Please put face_recognition model in path::{}'.format(face_recognition_model_filename))

        self.shape_predictor = dlib.shape_predictor(shape_predictor_filename)
        self.face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_filename)

    def fix_bbox(self, img, rect):
        bbox = _rect_to_bbox(rect)

        fixed_bbox = [None] * 4
        fixed_bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        fixed_bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        fixed_bbox[2] = img.shape[1] if bbox[2] > img.shape[1] else bbox[2]
        fixed_bbox[3] = img.shape[0] if bbox[3] > img.shape[0] else bbox[3]
        return fixed_bbox

    def extract_feature(self, img, show=False):
        has_face, package = self.get_face_locations(img) 

        if not has_face:
            if show:
                show_image(img, None, 'No face')
            return 
        else:
            rects, scores, idx = package
            rect_item = dict()
            for cur_rect, score, idx in zip(rects, scores, idx):
                size = cur_rect.bottom() - cur_rect.top()
                if not rect_item or size > rect_item['size']:
                    rect_item = {
                        'size': size,
                        'rect': cur_rect,
                        'score': score,
                        'direction': FaceDirection(idx).name,
                        'detect_obj': self.shape_predictor(img, cur_rect)
                    }

            if show:
                crop_img = dlib.get_face_chip(
                    img, rect_item['detect_obj'], size=320)
                cv2.imshow('crop-align', crop_img)

                text = 'Score: {:.5f}, Direction: {}'.format(
                    rect_item['score'], rect_item['direction'])
                bbox = self.fix_bbox(img, rect_item['rect'])
                show_image(img, bbox, text)

            return self.get_face_encoding(img, rect_item['detect_obj'])

    def get_face_locations(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects, scores, idx = self.face_detector.run(img_gray, 1)
        if len(rects) == 0:
            return False, None
        else:
            return True, (rects, scores, idx)

    def get_face_encoding(self, image, face_object_detect):
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        face_desc = self.face_recognition_model.compute_face_descriptor(
            image, face_object_detect)
        return np.array(face_desc)

class FaceModel:
    @classmethod
    def load_from_numpy(cls, filename, show):
        if not os.path.isfile(filename):
            print('Wrong pickle file path {}'.format(filename))
            exit()
        else:
            try:
                feature = np.load(filename)
                print('Model have {} samples'.format(feature.shape[0]))
                return cls(model=feature, show=show)

            except pickle.UnpicklingError as e:
                print('Wrong pickle format in file {}'.format(filename))
                exit()

    @classmethod
    def train_from_folder(cls, folder, save_model, show):
        if not os.path.isdir(folder):
            print('Wrong folder path {}'.format(folder))
            exit()
        else:
            return cls(folder=folder, save_model=save_model, show=show)

    def __init__(self, show=False, model=None, folder=None, save_model=None):
        self.show = show
        self.frtool = FaceExtractor()

        if model is not None:
            self.features = model

        elif folder is not None:
            self.features = self.train(folder, save_model)
        else:
            print('ERROR')
            exit()

    def train(self, folder, save_model):
        filenames = os.listdir(folder)

        features = list()
        project_root = os.getcwd()
        print('Get {} files'.format(len(filenames)))
        for filename in filenames:
            filepath = os.path.join(project_root, folder, filename)

            img = cv2.imread(filepath)
            if img is not None:
                feature = self.frtool.extract_feature(img, self.show)
                if feature is not None:
                    features.append(feature)
                else:
                    print('Cannot found face in {}'.format(filepath))
            else:
                print('Cannot read file {}'.format(filepath))

        if len(features) == 0:
            print('Cannot extract any feature from folder {}'.format(folder))
        else:
            features = np.array(features)
            num_samples = features.shape[0]
            print('Get {} samples'.format(num_samples))
            if save_model is None:
                np.save('model', features)
                print('Save to model.npy')
            else:
                np.save(save_model, features)
                print('save to {}'.format(save_model))
            return features

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
            dist = np.linalg.norm(self.features-feature, axis=1)
            print(np.mean(dist))

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
                    dist = np.linalg.norm(self.features-feature, axis=1)
                    avg_dist = np.mean(dist)
                    text = 'Similarity: {:.5f}'.format(avg_dist)
                    is_open = show_image(flip_frame, text=text, is_block=False)

            
def create_argparser():
    parser = argparse.ArgumentParser(description='FaceRecognition')

    mutual = parser.add_mutually_exclusive_group(required=True)
    mutual.add_argument('--load', action='store_true', default=False, help='Load from model file (*.npy)')
    mutual.add_argument('--train',action='store_true', default=False, help='Train model from images in folder')

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
        frtool = FaceModel.train_from_folder(args.input, args.output, args.show)
    elif args.load:
        frtool = FaceModel.load_from_numpy(args.input, args.show)

    frtool.predict_image(args.test)
    frtool.predict_from_stream(args.stream)