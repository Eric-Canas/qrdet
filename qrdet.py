"""
This class the YOLOv7 QR Detector. It uses a YOLOv7-tiny model trained to detect QR codes in the wild.

Author: Eric Canas.
Github: https://github.com/Eric-Canas/qrdet
Email: eric@ericcanas.com
Date: 11-12-2022
"""

from __future__ import annotations
import os
import requests
import tqdm
from yolov7_package import Yolov7Detector
import numpy as np

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '.yolov7_qrdet', 'qrdet-yolov7.pt')
_WEIGHTS_URL = 'https://github.com/Eric-Canas/qrdet/releases/download/first_qrdet_yolov7/qrdet-yolov7.pt'


class QRDetector:
    def __init__(self):
        """
        Initialize the QRDetector. It loads the weights of the YOLOv7 model and prepares it for inference.
        """
        if not os.path.exists(_WEIGHTS_PATH):
            self.__download_weights(url=_WEIGHTS_URL, path=_WEIGHTS_PATH)
        self.model = Yolov7Detector(weights=_WEIGHTS_PATH, img_size=None, agnostic_nms=True, traced=False)
        # Warm the model up.
        self.model.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray, return_confidences: bool = True, as_float: bool = False, is_bgr: bool = False) \
            -> tuple[tuple[list[float, float, float, float], float], ...] | tuple[list[float, float, float, float], ...]:
        """
        Detect QR codes in the given image.
        :param image: np.ndarray. The image to detect QR codes in, in RGB format.
        :param return_confidences: bool. Whether to return the confidences of the detections or not. Default: True.
        :param as_float: bool. Whether to return the bounding boxes as floats or not (int). Default: False.
        :param is_bgr: bool. Whether the image is in BGR format (True) or RGB format (False). Default: False.
        :return: tuple[tuple[list[float, float, float, float], float], ...] |
                    tuple[list[float, float, float, float], ...]. A tuple containing the bounding boxes of the QR codes
                    detected in the image, in the format ((x1, y1, x2, y2), ...). If return_confidences is True,
                    the tuple contains the confidence of the detection as well (((x1, y1, x2, y2), confidence),...).
        """
        # Check the image is in the correct format.
        assert type(image) is np.ndarray, f'Expected image to be a numpy array. Got {type(image)}.'
        assert image.dtype == np.uint8, f'Expected image to be of type np.uint8. Got {image.dtype}.'
        assert len(image.shape) == 3, f'Expected image to have 3 dimensions (H, W, RGB). Got {image.shape}.'
        assert image.shape[2] == 3, f'Expected image to have 3 channels (RGB). Got {image.shape[2]}.'
        # Transform the image from BGR to RGB if necessary (used when the image is read with OpenCV).
        if is_bgr:
            image = image[:, :, ::-1]
        dets = self.model.detect(image)
        # Check the detections return what it was expected (can be deleted after enough testing)
        assert all(len(detection) == 1 for detection in dets), 'Expected all detections to have 1 element.'
        assert all(detection == 0 for detection in dets[0][0]), 'Expected all detections to be of class 0 (QR code).'
        # Clip the bboxes to the image size and round them to int.
        h, w = image.shape[:2]
        bboxes = tuple(_clip_bbox(bbox=bbox, h=h, w=w, as_float=as_float) for bbox in dets[1][0])
        # Return the detections with or without the confidences.
        if return_confidences:
            return tuple((bbox, conf) for bbox, conf in zip(bboxes, dets[2][0]))
        else:
            return bboxes

    def __download_weights(self, url: str, path: str = _WEIGHTS_PATH,
                           desc: str = 'Downloading weights...') -> None:
        """
        Download the weights of the YOLOv7 model.
        :param url: str. The url of the weights.
        :param path: str. The path to save the weights. Default: _WEIGHTS_PATH.
        :param desc: str. The description of the download. Default: 'Downloading YOLOv7 QRDetector weights...'.
        """
        assert not os.path.isfile(path), f'Expected the weights to not exist. Found them at {path}.'
        # Create the directory to save the weights.
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Download the weights.
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=desc) as progress_bar:
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    file.write(data)

        # Check the weights were downloaded correctly.
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            # Delete the weights if the download failed.
            os.remove(path)
            raise Exception('Error, something went wrong while downloading the weights.')

def _clip_bbox(bbox: list[float, float, float, float], h: int, w: int, as_float:bool = False) -> \
                list[int | float, int | float, int | float, int | float]:
    """
    Clip the detected bbox to the image size. If as_float is False, round the bbox to int.

    :param bbox: list[int | float, int | float, int | float, int | float]. The detected bbox in format x1, y1, x2, y2.
                 as floats if as_float is True, as ints if as_float is False.
    :param h: int. The height of the image.
    :param w: int. The width of the image.
    :param as_float: bool. Whether to return the bbox as floats or not (int). Default: False (int).

    :return: list[int | float, int | float, int | float, int | float]. The clipped bbox in format x1, y1, x2, y2. As
                floats if as_float is True, as ints if as_float is False.
    """

    assert all(type(coord) is float for coord in bbox), 'Expected bbox to be a list of floats.'
    assert len(bbox) == 4, f'Expected bbox to have 4 elements. Got {len(bbox)}.'

    x1, y1, x2, y2 = bbox
    if as_float:
        x1, y1, x2, y2 = max(0., x1), max(0., y1), min(float(w), x2), min(float(h), y2)
    else:
        x1, y1, x2, y2 = max(0, round(x1)), max(0, round(y1)), min(w, round(x2)), min(h, round(y2))
    return [x1, y1, x2, y2]
