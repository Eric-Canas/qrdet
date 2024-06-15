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
    x1, y1, x2, y2 = detection['bbox_xyxy']
    confidence = detection['confidence']
    segmenation_xy = detection['quadrilateral_xy']
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)
# Save the results
cv2.imwrite(filename='resources/qreader_test_image_detections.jpeg', img=image)
```

<img alt="detections_output" title="detections_output" src="https://raw.githubusercontent.com/Eric-Canas/qrdet/main/resources/qreader_test_image_detections.jpeg" width="100%">

## API Reference

### QRDetector(model_size = 's', conf_th = 0.5, nms_iou = 0.3, weights_folder = '<qrdet_package>/.model')

- ``model_size``: **"n"|"s"|"m"|"l"**. Size of the model to load. Smaller models will be faster, while larger models will be more capable for difficult situations. Default: `'s'`.
- ``conf_th``: **float**. Confidence threshold to consider that a detection is valid. Incresing this value will reduce _false positives_ while decreasing will reduce _false_negatives_. Default: `0.5`.
- ``nms_iou``: **float**. _Intersection over Union_ (IoU) threshold for _Non-Maximum Suppression_ (NMS). NMS is a technique used to eliminate redundant bounding boxes for the same object. Increase this number if you find problems with duplicated detections. Default: `0.3`
- ``weights_folder``: **str**. Folder where detection model will be downloaded. By default, it points out to an internal folder within the package, making sure that it gets correctly removed when uninstalling. You could need to change it when working in environments like [AWS Lambda](https://aws.amazon.com/es/pm/lambda/) where only [/tmp folder is writable](https://docs.aws.amazon.com/lambda/latest/api/API_EphemeralStorage.html), as issued in [#11](https://github.com/Eric-Canas/qrdet/issues/11). Default: `'<qrdet_package>/.model'`.

### QRDetector.detect(image, is_bgr = False, **kwargs)

- ``image``: **np.ndarray|'PIL.Image'|'torch.Tensor'|str**. `np.ndarray` of shape **(H, W, 3)**, `PIL.Image`, `Tensor` of shape **(1, 3, H, W)**, or `path`/`url` to the image to predict. `'screen'` for grabbing a screenshot.
- ``is_bgr``: **bool**. If `True` the image is expected to be in **BGR**. Otherwise, it will be expected to be **RGB**. Only used when image is `np.ndarray` or `torch.tensor`. Default: `False`
- ``legacy``: **bool**. If sent as **kwarg**, will parse the output to make it identical to 1.x versions. Not Recommended. Default: False.

- **Returns**: **tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]**. A tuple of dictionaries containing all the information of every detection. Contains the following keys.

| Key              | Value Desc.                                 | Value Type                 | Value Form                  |
|------------------|---------------------------------------------|----------------------------|-----------------------------|
| `confidence`     | Detection confidence                        | `float`                    | `conf.`                     |
| `bbox_xyxy`      | Bounding box                                | np.ndarray (**4**)         | `[x1, y1, x2, y2]`          |
| `cxcy`           | Center of bounding box                      | tuple[`float`, `float`]    | `(x, y)`                    |
| `wh`             | Bounding box width and height               | tuple[`float`, `float`]    | `(w, h)`                    |
| `polygon_xy`     | Precise polygon that segments the _QR_      | np.ndarray (**N**, **2**)  | `[[x1, y1], [x2, y2], ...]` |
| `quad_xy`        | Four corners polygon that segments the _QR_ | np.ndarray (**4**, **2**)  | `[[x1, y1], ..., [x4, y4]]` |
| `padded_quad_xy` |`quad_xy` padded to fully cover `polygon_xy` | np.ndarray (**4**, **2**)  | `[[x1, y1], ..., [x4, y4]]` |
| `image_shape`    | Shape of the input image                    | tuple[`float`, `float`]    | `(h, w)`                    |  

> **NOTE:**
> - All `np.ndarray` values are of type `np.float32` 
> - All keys (except `confidence` and `image_shape`) have a normalized ('n') version. For example,`bbox_xyxy` represents the bbox of the QR in image coordinates [[0., im_w], [0., im_h]], while `bbox_xyxyn` contains the same bounding box in normalized coordinates [0., 1.].
> - `bbox_xyxy[n]` and `polygon_xy[n]` are clipped to `image_shape`. You can use them for indexing without further management

## Acknowledgements

This library is based on the following projects:

- <a href="https://github.com/ultralytics/ultralytics" target="_blank">YoloV8</a> model for **Object Segmentation**.
- <a href="https://github.com/Eric-Canas/quadrilateral-fitter" target="_blank">QuadrilateralFitter</a> for fitting 4 corners polygons from noisy **segmentation outputs**.
