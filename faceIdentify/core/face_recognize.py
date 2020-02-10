# coding: utf-8
import os
import pickle

import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

from core.face_extractor import FaceExtractor

class FaceIdentify:
    def __init__(self, show):
        self.frtool = FaceExtractor()
        self.show = show

    def load(self, model_file):
        try:
            feature = np.load(filename)
            print('Model have {} samples'.format(feature.shape[0]))
            self.model = feature

        except IOError:
            print('Wrong numpy format in file {}'.format(filename))
            exit()

    def identify(self, feature):
        dist = np.linalg.norm(self.model-feature, axis=1)
        return np.mean(dist)

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
            self.model = features

class FaceClassifier:
    def __init__(self):
        self.frtool = FaceExtractor()

    def recog(self, feature):
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        return self.model.predict(feature)

    def recog_proba(self, feature):
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        proba = self.model.predict_proba(feature)
        identified = self.model.classes_[np.argmax(proba)]

        return identified, proba

    def load(self, model_file):
        try:
            self.model = pickle.load(open(model_file, 'rb'))
        except pickle.UnpicklingError as e:
            print('Wrong pickle file {}'.format(model_file))

    def train(self, root, save_model):
        subfolders = {
            folder: os.path.join(root, folder)
            for folder in os.listdir(root)
        }

        users_encoding = dict()
        for _id, folder in subfolders.items():
            if not os.path.isdir(folder):
                print('Not folder, skip {}'.format(folder))
            else:
                print('Doing {}'.format(folder))
                files = [os.path.join(folder, _file)
                for _file in os.listdir(folder)]
                
                user_encoding = list()
                for _file in filter(os.path.isfile, files):
                    img = cv2.imread(_file)
                    if img is not None:
                        feature = self.frtool.extract_feature(img)
                        if feature is not None:
                            user_encoding.append(feature)
                
                if len(user_encoding):
                    users_encoding[_id] = user_encoding
                print('Finish {}'.format(folder))

        X = np.concatenate(list(users_encoding.values()))
        Y = np.array(list(_id for _id, encodings in users_encoding.items() for loop in range(len(encodings))))

        clf = KNeighborsClassifier(algorithm='ball_tree')
        clf.fit(X,Y)

        self.model = clf

        pickle.dump(clf, open(save_model, 'wb'))