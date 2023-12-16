import cv2

from pathlib import Path
from deepface import DeepFace

class FaceRecognition:
    """
    A wrapper for DeepFace face recognition model.
    """
    def __init__(self, detector_backend='dlib'):
        self.detector_backend = detector_backend

    def get_faces(self, im):
        preds = DeepFace.extract_faces(im, detector_backend=self.detector_backend)
        # No faces detected
        if len(preds) == 0:
            return None
        return [{'face' : pred['face'], 'bounding_box': self._pack_bounding_box(pred['facial_area'])} for pred in preds]

    def find_faces(self, im, database):
        preds = self.get_faces(im)
        # Image does not contain any visible face
        if len(preds) == 0:
            return []

        result = []
        for pred in preds:
            df = DeepFace.find(
                (pred['face'] * 255).astype('uint8'), 
                db_path=str(database),
                enforce_detection=False,
                detector_backend=self.detector_backend
            )

            if df[0].empty:
                result.append(('Unknown', pred['bounding_box']))
            else:
                face_id = self._unpack_top_face_id(df)
                face_label = database[face_id]
                result.append((face_label, pred['bounding_box']))
        
        return result

    def _unpack_top_face_id(self, df):
        return Path(df[0]['identity'].iloc[0]).stem

    def _pack_bounding_box(self, facial_area):
        return facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']