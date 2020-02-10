# coding: utf-8
import enum
import os

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
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # rects, scores, idx = self.face_detector.run(img_gray, 1)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects, scores, idx = self.face_detector.run(img, 0)

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