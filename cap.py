import os
from omegaconf import DictConfig
import torch
from PIL import Image
from grit.datasets.caption.field import TextField
from grit.datasets.caption.transforms import get_transform
from grit.engine.utils import nested_tensor_from_tensor_list
from grit.models.caption import Transformer
from grit.models.caption.detector import build_detector
from hydra import compose, initialize
from omegaconf import OmegaConf

from gtts import gTTS
from IPython.display import Audio
from googletrans import Translator
import cv2
import uuid
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from deepface import DeepFace

from lib.config import config
from lib.face_database import FaceDatabase
from lib.face_recognition import FaceRecognition

face_database = FaceDatabase(**config.face_database)
face_recognition = FaceRecognition(**config.face_recognition)

list_of_lang = 'https://gtts.readthedocs.io/en/v2.2.0/_modules/gtts/lang.html'


class Caption:
    def __init__(
        self,
        tts_path,
        voc_path,
        out_path,
        exp_checkpoint
    ):
        self.tts_path = tts_path
        self.voc_path = voc_path
        self.out_path = out_path
        self.exp_checkpoint = exp_checkpoint

        initialize(config_path="grit/configs/caption", job_name="inference_caption")
        self.config = compose(config_name="coco_config")#, overrides=[
        #    f"img_path={'example.jpg'}",
        #    f"vocab_path={self.voc_path}",
        #    f"exp.checkpoint={self.exp_checkpoint}"
        #])

        self.device = torch.device("cuda:0")
        self.detector = build_detector(self.config).to(self.device)
        self.model = Transformer(detector=self.detector, config=self.config)
        self.model = self.model.to(self.device)

        # load checkpoint
        if os.path.exists(self.config.exp.checkpoint):
            self.checkpoint = torch.load(self.config.exp.checkpoint, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(self.checkpoint['state_dict'], strict=False)
            #print(f"model missing:{len(missing)} model unexpected:{len(unexpected)}")

        self.model.cached_features = False

        # prepare utils
        self.transform = get_transform(self.config.dataset.transform_cfg)['valid']
        self.text_field = TextField(vocab_path=self.config.vocab_path if 'vocab_path' in self.config else self.config.dataset.vocab_path)

    def inference_caption(self, img_path):
        # load image
        rgb_image = Image.open(img_path).convert('RGB')
        image = self.transform(rgb_image)
        images = nested_tensor_from_tensor_list([image]).to(self.device)

        # inference and decode
        with torch.no_grad():
            out, _ = self.model(
                images,
                seq=None,
                use_beam_search=True,
                max_len=self.config.model.beam_len,
                eos_idx=self.config.model.eos_idx,
                beam_size=self.config.model.beam_size,
                out_size=1,
                return_probs=False,
            )
            caption = self.text_field.decode(out, join_words=True)[0]
            print(f'Generated Caption: {caption}')
            with open(self.out_path, 'w') as f:
                f.write(caption)

    def convert_to_mp3(
        self,
        text,
        lang = 'en', 
        filename ='output'
    ):
        tts = gTTS(text=text, lang=lang)
        tts.save(self.tts_path + filename)
        Audio(self.tts_path + filename)
    
    def translate_and_convert(self,text, target_language = 'en', filename = 'output'):
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        self.convert_to_mp3(translation.text,target_language, filename)
        return translation.text

class FaceRecognition:
    def __init__(self):
        self.face_database = FaceDatabase(**config.face_database)
        self.face_recognition = FaceRecognition(**config.face_recognition)
    def append_database(self,img_path):
        im = cv2.imread(f'{img_path}')

            # 2. Detect faces
        detected = self.face_recognition.get_faces(im)

            # 3. For each face in the image, query label
        labels = []
        for it, individual in enumerate(detected):
            plt.figure(it, [5, 5]); plt.imshow(individual['face']); plt.axis('off')
            labels.append(input())
        save = input('Save ?')
        if(save):
            for individual, label in zip(detected, labels):
                self.face_database.store(individual['face'], label)
                individual['face'] = cv2.resize(individual['face'], (224,224))
    def query(self,img_path):
        # 1. Get query image (possibly with no or multiple faces)
        query = cv2.imread(f'{img_path}').astype('uint8')

        # 2. Find matches for each face
        detected = face_recognition.find_faces(query, self.face_database)

        # 3. Show matches
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        self.labels = []
        for label, bounding_box in detected:
            x, y, w, h = bounding_box
            cv2.rectangle(query, (x, y), (x+w, y+h), (0,0,255), 2)
            print(f'Detected: {label} (at position ({x}, {y}))')
            self.labels.append(label)
        plt.figure(1, [5, 5]); plt.title('Result'); plt.imshow(query)    

if __name__ == '__main__':
    caption_generator = Caption(
        tts_path='',
        voc_path='grit/data/vocab.json',
        out_path='./output.txt',
        exp_checkpoint='grit/grit_checkpoint_vg.pth'
    )

    caption_generator.inference_caption('example.jpg')