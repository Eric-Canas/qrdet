# QRDet
**QRDet** is a robust **QR Detector** based on <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8</a>.

**QRDet** will detect & segment **QR** codes even in **difficult** positions or **tricky** images. If you are looking for a complete **QR Detection** + **Decoding** pipeline, take a look at <a href="https://github.com/Eric-Canas/qreader" target="_blank">QReader</a>.  

## Installation

To install **QRDet**, simply run:

```bash
pip install qrdet
```

## Usage

There is only one function you'll need to call to use **QRDet**, ``detect``:

```python

from qrdet import QRDetector
import cv2

detector = QRDetector(model_size='s')
image = cv2.imread(filename='resources/qreader_test_image.jpeg')
detections = detector.detect(image=image, is_bgr=True)

# Draw the detections
for detection in detections:
    x1, y1, x2, y2 = detections['bbox_xyxy']
    confidence = detections['confidence']
    segmenation_xy = detections['quadrilateral_xy']
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)
# Save the results
cv2.imwrite(filename='resources/qreader_test_image_detections.jpeg', img=image)
```

<img alt="detections_output" title="detections_output" src="https://raw.githubusercontent.com/Eric-Canas/qrdet/main/resources/qreader_test_image_detections.jpeg" width="100%">

## API Reference

### QReader.detect(image, is_bgr = False, **kwargs)

- ``image``: **np.ndarray|'PIL.Image'|'torch.Tensor'|str**. `np.ndarray` of shape **(H, W, 3)**, `PIL.Image`, `Tensor` of shape **(1, 3, H, W)**, or `path`/`url` to the image to predict. `'screen'` for grabbing a screenshot.
- ``is_bgr``: **bool**. If `True` the image is expected to be in **BGR**. Otherwise, it will be expected to be **RGB**. Only used when image is `np.ndarray` or `torch.tensor`. Default: `False`
- ``legacy``: **bool**. If sent as **kwarg**, will parse the output to make it identical to 1.x versions. Not Recommended. Default: False.

- **Returns**: **tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]**. A tuple of dictionaries containing the following keys:
    - `confidence`: **float**. The confidence of the detection.
    - `bbox_xyxy`: **np.ndarray**. The bounding box of the detection in the format **(x1, y1, x2, y2)**, dtype: `np.float32`.
    - `cxcy`: **tuple[float, float]**. The center of the bounding box in the format **(x, y)**.
    - `wh`: **tuple[float, float]**. The width and height of the bounding box in the format **(w, h)**.
    - `polygon_xy`: **np.ndarray**. The accurate polygon that surrounds the QR code, with shape **(N, 2)**.
    - `quadrilateral_xy`: **np.ndarray**. The quadrilateral that surrounds the QR code, with shape **(4, 2)**, dtype: `np.float32`.
    - `expanded_quadrilateral_xy`: **np.ndarray**. An expanded version of quadrilateral_xy, with shape **(4, 2)**, dtype: `np.float32`, that always include all the points within `'polygon_xy'`.
    - `image_shape`: **tuple[int, int]**. Shape of the input image, in the format **(h, w)**.

All these keys (except `'confidence'` and `'image_shape'`) have a `'n'` (_normalized_) version. For example, `'bbox_xyxy'` is the bounding box in **absolute coordinates**, while `'bbox_xyxyn'` is the bounding box in **normalized** coordinates (from 0. to 1.).

## Acknowledgements

This library is based on the following projects:

- <a href="https://github.com/ultralytics/ultralytics" target="_blank">YoloV8</a> model for **Object Segmentation**.
- <a href="https://github.com/Eric-Canas/quadrilateral-fitter" target="_blank">QuadrilateralFitter</a> for fitting 4 corners polygons from noisy **segmentation outputs**.
