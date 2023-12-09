import os
from yacs.config import CfgNode as CN
from pathlib import Path

config = CN()

config.face_database = CN()
config.face_database.db_root = "./database/"
config.face_database.im_ext = ".jpg"
config.face_database.im_dim = (224, 224)

config.face_recognition = CN()
config.face_recognition.detector_backend = "retinaface"