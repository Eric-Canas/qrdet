"""
This class the YOLOv7 QR Detector. It uses a YOLOv7-tiny model trained to detect QR codes in the wild.

Author: Eric Canas.
Github: https://github.com/Eric-Canas/qrdet
Email: eric@ericcanas.com
Date: 11-12-2022
"""

from __future__ import annotations
import os

import numpy as np
import requests
import tqdm

from ultralytics import YOLO

from qrdet import _yolo_v8_results_to_dict, _prepare_input, BBOX_XYXY, CONFIDENCE

_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), '.model')
_CURRENT_RELEASE_TXT_FILE = os.path.join(_WEIGHTS_FOLDER, 'current_release.txt')
_WEIGHTS_URL_FOLDER = 'https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release'
_MODEL_FILE_NAME = 'qrdet-{size}.pt'


class QRDetector:
    def __init__(self, model_size: str = 's', conf_th: float = 0.5, nms_iou: float = 0.3,
                 weights_folder: str = _WEIGHTS_FOLDER):
        """
        Initialize the QRDetector.
        It loads the weights of the YOLOv8 model and prepares it for inference.
        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param conf_th: float. The confidence threshold to use for the detections. Detection with a confidence lower
                                than this value will be discarded. Default: 0.5.
        :param nms_iou: float. The IoU threshold to use for the Non-Maximum Suppression. Detections with an IoU higher
                                than this value will be discarded. Default: 0.3.
        """
        assert model_size in ('n', 's', 'm', 'l'), f'Invalid model size: {model_size}. ' \
                                                   f'Valid values are: \'n\', \'s\', \'m\' or \'l\'.'
        self._model_size = model_size
        self.weights_folder = weights_folder
        self.__current_release_txt_file = os.path.join(weights_folder, 'current_release.txt')

        path = self.__download_weights_or_return_path(model_size=model_size)
        assert os.path.exists(path), f'Could not find model weights at {path}.'

        self.model = YOLO(model=path, task='segment')

        self._conf_th = conf_th
        self._nms_iou = nms_iou

    def detect(self, image: np.ndarray|'PIL.Image'|'torch.Tensor'|str, is_bgr: bool = False,
               **kwargs) -> tuple[dict[str, np.ndarray|float|tuple[float, float]]]:
        """
        Detect QR codes in the given image.

        :param image: str|np.ndarray|PIL.Image|torch.Tensor. Numpy array (H, W, 3), Tensor (1, 3, H, W), or
                                            path/url to the image to predict. 'screen' for grabbing a screenshot.
        :param legacy: bool. If sent as **kwarg**, will parse the output to make it identical to 1.x versions.
                            Not Recommended. Default: False.
        :return: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
            - 'expanded_quadrilateral_xy': np.ndarray. An expanded version of quadrilateral_xy, with shape (4, 2),
                that always include all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        image = _prepare_input(source=image, is_bgr=is_bgr)
        # Predict
        results = self.model.predict(source=image, conf=self._conf_th, iou=self._nms_iou, half=False,
                                device=None, max_det=100, augment=False, agnostic_nms=True,
                                classes=None, verbose=False)
        assert len(results) == 1, f'Expected 1 result if no batch sent, got {len(results)}'
        results = _yolo_v8_results_to_dict(results = results[0], image=image)

        if 'legacy' in kwargs and kwargs['legacy']:
            return self._parse_legacy_results(results=results, **kwargs)
        return results



    def _parse_legacy_results(self, results, return_confidences: bool = True, **kwargs) \
            -> tuple[tuple[list[float, float, float, float], float], ...] | tuple[list[float, float, float, float], ...]:
        """
        Parse the results to make it compatible with the legacy version of the library.
        :param results: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. The results to parse.
        """
        if return_confidences:
            return tuple((result[BBOX_XYXY], result[CONFIDENCE]) for result in results)
        else:
            return tuple(result[BBOX_XYXY] for result in results)


    def __download_weights_or_return_path(self, model_size: str = 's', desc: str = 'Downloading weights...') -> None:
        """
        Download the weights of the YoloV8 QR Segmentation model.
        :param model_size: str. The size of the model to download. Can be 's', 'm' or 'l'. Default: 's'.
        :param desc: str. The description of the download. Default: 'Downloading weights...'.
        """
        self.downloading_model = True
        path = os.path.join(self.weights_folder, _MODEL_FILE_NAME.format(size=model_size))
        if os.path.isfile(path):
            if os.path.isfile(self.__current_release_txt_file):
                # Compare the current release with the actual release URL
                with open(self.__current_release_txt_file, 'r') as file:
                    current_release = file.read()
                # If the current release is the same as the URL, the weights are already downloaded.
                if current_release == _WEIGHTS_URL_FOLDER:
                    self.downloading_model = False
                    return path
        # Create the directory to save the weights.
        elif not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

        url = f"{_WEIGHTS_URL_FOLDER}/{_MODEL_FILE_NAME.format(size=model_size)}"

        # Download the weights.
        from warnings import warn
        warn("QRDetector has been updated to use the new YoloV8 model. Use legacy=True when calling detect "
             "for backwards compatibility with 1.x versions. Or update to new output (new output is a tuple of dicts, "
             "containing several new information (1.x output is accessible through 'bbox_xyxy' and 'confidence')."
             "Forget this message if you are reading it from QReader. [This is a first download warning and will be removed at 2.1]")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=desc) as progress_bar:
            with open(path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    file.write(data)
        # Save the current release URL
        with open(self.__current_release_txt_file, 'w') as file:
            file.write(_WEIGHTS_URL_FOLDER)
        # Check the weights were downloaded correctly.
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            # Delete the weights if the download failed.
            os.remove(path)
            raise EOFError('Error, something went wrong while downloading the weights.')

        self.downloading_model = False
        return path

    def __del__(self):
        if hasattr(self, 'weights_folder'):
            path = os.path.join(self.weights_folder, _MODEL_FILE_NAME.format(size=self._model_size))
            # If the weights didn't finish downloading, delete them.
            if hasattr(self, 'downloading_model') and self.downloading_model and os.path.isfile(path):
                os.remove(path)