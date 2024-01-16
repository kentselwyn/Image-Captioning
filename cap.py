import os

import torch
import numpy as np

from PIL import Image
from grit.datasets.caption.field import TextField
from grit.datasets.caption.transforms import get_transform
from grit.engine.utils import nested_tensor_from_tensor_list
from grit.models.caption import Transformer
from grit.models.caption.detector import build_detector
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

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

from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

from lib.config import config
from lib.face_database import FaceDatabase
from lib.face_recognition import FaceRecognition

import nltk
from nltk import Tree
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

from scipy.ndimage import filters
from skimage import transform as skimage_transform

import matplotlib.pyplot as plt

list_of_lang = 'https://gtts.readthedocs.io/en/v2.2.0/_modules/gtts/lang.html'

class ImageCaptioningWithFaceRecognition:
    def __init__(
        self,
        tts_path,
        out_path,
        exp_checkpoint,
        face_recognition_config
    ):
        self.tts_path = tts_path
        self.out_path = out_path
        self.exp_checkpoint = exp_checkpoint

        initialize(config_path="grit/configs/caption", job_name="inference_caption")
        self.config = compose(config_name="coco_config")

        self.device = torch.device("cuda:0")
        self.detector = build_detector(self.config).to(self.device)
        self.model = Transformer(detector=self.detector, config=self.config)
        self.model = self.model.to(self.device)

        # Load checkpoint
        if os.path.exists(self.exp_checkpoint):
            print(f"Loading checkpoint {self.exp_checkpoint}...")
            self.checkpoint = torch.load(self.exp_checkpoint, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(self.checkpoint["state_dict"], strict=False)

        self.model.cached_features = False

        # Prepare utils
        self.transform = get_transform(self.config.dataset.transform_cfg)['valid']
        self.text_field = TextField(vocab_path=self.config.vocab_path if 'vocab_path' in self.config else self.config.dataset.vocab_path)

        # Prepare face recognition and database
        self.face_database = FaceDatabase(**face_recognition_config.face_database)
        self.face_recognition = FaceRecognition(**face_recognition_config.face_recognition)

        # Prepare text localization tools
        self.text_local_model, self.text_local_vis_processors, self.text_local_text_processors = load_model_and_preprocess(
            name="blip_image_text_matching",
            model_type="base",
            device="cpu",
            is_eval=True
        )

    def inference_caption(self, img_path):
        # Load image
        rgb_image = Image.open(img_path).convert("RGB")
        image = self.transform(rgb_image)
        images = nested_tensor_from_tensor_list([image]).to(self.device)

        # Inference and decode
        caption = None
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
            with open(self.out_path, "w") as f:
                f.write(caption)
        print(f"Initially generated caption: {caption}")
        
        img = (np.asarray(rgb_image)[..., :3]).astype(np.float32)
        print(f"Image shape: {img.shape}, RGB Image shape: {rgb_image.size}")
        faces = self.face_recognition.find_faces(
            im=img_path, 
            database=self.face_database
        )
        # Remove unknowns
        faces = [face for face in faces if face[0] != "Unknown"]
        if faces == []:
            print(f"No face detected.\nGenerated caption: {caption}")
            return
        # Compute mask for every face
        mask = np.zeros((len(faces), *img.shape[:2]), dtype=np.uint8)
        for it, tp in enumerate(faces):
            _, bounding_box = tp
            y, x, w, h = bounding_box
            mask[it, x:x+h, y:y+w] = 1
        # Parse caption to nltk grammar tree
        img_features = self.text_local_vis_processors["eval"](rgb_image).unsqueeze(0).to("cpu")
        txt = self.text_local_text_processors["eval"](caption)
        txt_tokens = self.text_local_model.tokenizer(txt, return_tensors="pt").to("cpu")
        decoded_caption = []
        for token_id in txt_tokens.input_ids[0][1:-1]:
            decoded_caption.append(self.text_local_model.tokenizer.decode([token_id]))
        decoded_caption = ' '.join(decoded_caption)
        tokenized = nltk.sent_tokenize(decoded_caption)
        words = nltk.word_tokenize(tokenized[0])
        tagged_words = nltk.pos_tag(words)
        chunk_grammar = r"""
        NBAR:
            {<CD|NN.*|JJ|DT|\#>*<NN.*>}
        NP:
            {<NBAR><IN><NBAR>}
            {<NBAR>}
        """
        chunk_parser = nltk.RegexpParser(chunk_grammar)
        chunked = chunk_parser.parse(tagged_words)
        # Create tree index to list index mapping
        leaf2index_map = {}
        for idx, _ in enumerate(chunked.leaves()):
            leaf2index_map[chunked.leaf_treeposition(idx)] = idx
        # Create noun phrase ID to tree index mapping
        ID2tree_map = {}
        # Assign each noun phrase into unique IDs
        group_ID = -1
        noun_phrase_group = [-1] * len(chunked.leaves())
        prev_group_prefix = None
        for it, tree_position in enumerate(chunked.treepositions()):
            if not isinstance(chunked[tree_position], tuple) and \
               chunked[tree_position].label() == "NP":
               prev_group_prefix = tree_position
               group_ID += 1
               ID2tree_map[group_ID] = tree_position
            if tree_position is leaf2index_map.keys() and \
               prev == tree_position[:len(prev)]:
               noun_phrase_group[leaf2index_map[tree_position]] = group_ID
        # Confidence in replace
        confidence = np.zeros((len(faces), group_ID + 1), dtype=np.uint32) - 1
        # Compute gradcam
        gradcam, _ = compute_gradcam(
            self.text_local_model,
            img_features,
            txt,
            txt_tokens,
            block_num=7,
        )
        gradcam_iter = iter(gradcam[0][2:-1])
        token_id_iter = iter(txt_tokens.input_ids[0][1:-1])
        for it, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
            word = self.text_local_model.tokenizer.decode([token_id])
            tag = nltk.pos_tag([word])[0][1]
            # Only consider nouns
            if 'NN' in tag:
                gradcam = gradcam - gradcam.min()
                if gradcam.max() > 0:
                    gradcam = gradcam / gradcam.max()
                gradcam = skimage_transform.resize(
                    gradcam.cpu().numpy(),
                    img.shape[:2],
                    order=3,
                    mode="constant"
                )
                gradcam = filters.gaussian_filter(gradcam, 0.02 * max(img_features.shape[:-2]))
                gradcam = (gradcam - gradcam.min()) / (gradcam.max())
                gradcam[gradcam > 0.4] = 1
                # Compute score
                for face_it in range(len(faces)):
                    score = np.sum(gradcam * mask[face_it])
                    confidence[face_it, noun_phrase_group[it]] = np.maximum(confidence[face_it, noun_phrase_group[it]], score)
        # Replace noun phrases
        to_replace = np.argmax(confidence, axis=1)
        for replaced_groupID in np.unique(to_replace):
            collective_labels = []
            for face_ID in np.where(to_replace == replaced_groupID)[0]:
                collective_labels.append(faces[face_ID][0])
            new_noun_phrase = None
            if len(collective_labels) > 1:
                new_noun_phrase = ', '.join(collective_labels[:-1])
                new_noun_phrase += ' and ' + collective_labels[-1]
            else:
                new_noun_phrase = collective_labels[0]
            new_tagged_words = nltk.pos_tag([new_noun_phrase])
            new_chunked = chunk_parser.parse(new_tagged_words)
            chunked[int(replaced_groupID)] = new_chunked

        new_caption = " ".join([w for w, t in chunked.leaves()])
        new_caption.replace("# ", "")
        print(f"Generated caption: {new_caption}")

    def convert_to_mp3(
        self,
        text,
        lang="en", 
        filename="output"
    ):
        tts = gTTS(text=text, lang=lang)
        tts.save(self.tts_path + filename)
        Audio(self.tts_path + filename)
    
    def translate_and_convert(self,text, target_language = 'en', filename = 'output'):
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        self.convert_to_mp3(translation.text,target_language, filename)
        return translation.text
    
    def detect_faces(self, img=None, img_path=None):
        # Sanity check
        assert (img is not None) or (img_path is not None), "No input image or path is given!"
        if img_path is not None:
            img = cv2.imread(img_path)
        detected = self.face_recognition.get_faces(img)
        faces = []
        for individual in detected:
            faces.append(individual["face"])
        return faces

    def insert_faces_and_names(self, faces, names):
        for face, name in zip(faces, names):
            self.face_database.store(face, name)  

if __name__ == "__main__":
    caption_generator = ImageCaptioningWithFaceRecognition(
        tts_path="",
        out_path="./output.txt",
        exp_checkpoint="grit_checkpoint_vg.pth",
        face_recognition_config=config
    )

    faces = caption_generator.detect_faces(img_path="img/bill_2.jpg")
    names = ["Bill"]
    caption_generator.insert_faces_and_names(faces, names)

    #faces = caption_generator.detect_faces(img_path="img/felix.jpg")
    #names = ["Felix"]
    #caption_generator.insert_faces_and_names(faces, names)

    #faces = caption_generator.detect_faces(img_path="img/samuel.jpg")
    #names = ["Samuel"]
    #caption_generator.insert_faces_and_names(faces, names)

    #faces = caption_generator.detect_faces(img_path="img/angie.jpg")
    #names = ["Angeline"]
    #caption_generator.insert_faces_and_names(faces, names)

    #faces = caption_generator.detect_faces(img_path="img/kent.jpg")
    #names = ["Kent"]
    #caption_generator.insert_faces_and_names(faces, names)

    caption_generator.inference_caption(img_path="img/test.jpg")