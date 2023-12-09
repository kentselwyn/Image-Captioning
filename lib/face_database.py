import os
import cv2
import uuid
import pickle
import pandas as pd

from pathlib import Path

class FaceDatabase:
    """
    Database management class for face recognition.
    FaceDatabase facilitates storing and quering images and labels.
    """
    def __init__(self, db_root, im_ext=".jpg", im_dim=(224,224)):
        self.db_root = db_root
        self.im_ext = im_ext
        self.im_dim = im_dim

        # Load face ID table
        self.face_id = {}
        if os.path.isfile(self.db_root + "/id"):
            self.face_id = pd.read_pickle(self.db_root + "/id")

        # Sanity check
        if not os.path.isdir(self.db_root):
            os.mkdir(self.db_root)
        assert os.path.isdir(self.db_root), "Database root does not exist"
    
    def store(self, im, label):
        im_fname = self._generate_filename()

        # Update face ID table
        self.face_id[im_fname] = label

        # Write image to database directory
        im = cv2.resize(im, self.im_dim)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.db_root + im_fname + self.im_ext, 255*im)

        # Remove cache if exists
        if os.path.isfile(self.db_root + "/representations_vgg_face.pkl"):
            os.remove(self.db_root + "/representations_vgg_face.pkl")

    def _generate_filename(self):
        # Generate unique filename
        return uuid.uuid4().hex

    def __del__(self):
        with open(self.db_root + "/id", "wb") as fh:
            pickle.dump(self.face_id, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        return self.db_root

    def __getitem__(self, im_fname):
        if im_fname not in self.face_id.keys():
            return None
        return self.face_id[im_fname]