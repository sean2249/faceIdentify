# coding: utf-8
from performance import FaceIdentify

class FaceRecognize:
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

            except IOError:
                print('Wrong numpy format in file {}'.format(filename))
                exit()

    @classmethod
    def train_from_folder(cls, folder, save_model, show):
        if not os.path.isdir(folder):
            print('Wrong folder path {}'.format(folder))
            exit()
        else:
            # return cls(folder=folder, save_model=save_model, show=show)

    def __init__(self, show=False, model=None, folder=None, save_model=None):
        self.show = show
        self.frtool = FaceExtractor()
        self.frclassifier = FaceClassifier(model)

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
            dist = self.frclassifier.identify(feature)
            print(dist)
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
                    avg_dist = self.frclassifier.identify(feature)
                    # print(dist)
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
