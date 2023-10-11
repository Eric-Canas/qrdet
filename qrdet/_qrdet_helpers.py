from __future__ import annotations

from ultralytics.engine.results import Results

from qrdet import BBOX_XYXY, BBOX_XYXYN, POLYGON_XY, POLYGON_XYN,\
    CXCY, CXCYN, WH, WHN, IMAGE_SHAPE, CONFIDENCE, PADDED_QUAD_XY, PADDED_QUAD_XYN,\
    QUAD_XY, QUAD_XYN

from quadrilateral_fitter import QuadrilateralFitter
import numpy as np
from math import floor, ceil

def _yolo_v8_results_to_dict(results: Results, image: np.ndarray) -> \
        tuple[dict[str, np.ndarray|float|tuple[float, float]]]:
    """
    Converts a Results object from YOLOv8 to a list of dictionaries. Each dictionary
    contains the polygon and bounding box coordinates for a single detection.

    :param results: ultralytics.Results. Results object from YOLOv8
    :return: :return: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2). This
                quadrilateral is fitted to make sure that all points in polygon_xy are contained inside.
            - 'quadrilateral_xy': np.ndarray. A tighter and more accurate version of quadrilateral_xy, with shape
                (4, 2). It's more adjusted to QR code borders, but it may not contain all points in polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
    """
    if len(results) == 0:
        return []
    im_h, im_w = results.orig_shape[:2]
    detections = []
    results = results.cpu()
    for result in results:
        boxes = result.boxes.numpy()
        assert len(boxes) == 1, f'Expected boxes result to have length 1, got {len(result)}'
        bbox_xyxy, bbox_xyxyn = boxes.xyxy[0], boxes.xyxyn[0]

        mask = result.masks
        assert len(mask) == 1, f'Expected mask result to have length 1, got {len(result)}'
        accurate_polygon_xy, accurate_polygon_xyn = mask.xy[0], mask.xyn[0]
        # Fit a quadrilateral to the polygon (Don't clip accurate_polygon_xy yet, to fit the quadrilateral before)
        _quadrilateral_fit = QuadrilateralFitter(polygon=accurate_polygon_xy)
        quadrilateral_xy = _quadrilateral_fit.fit(simplify_polygons_larger_than=8,
                                                  start_simplification_epsilon=0.1,
                                                  max_simplification_epsilon=2.,
                                                  simplification_epsilon_increment=0.2)
        # Clip the data to make sure it's inside the image
        np.clip(bbox_xyxy[::2], a_min=0., a_max=im_w, out=bbox_xyxy[::2])
        np.clip(bbox_xyxy[1::2], a_min=0., a_max=im_h, out=bbox_xyxy[1::2])
        np.clip(bbox_xyxyn, a_min=0., a_max=1., out=bbox_xyxyn)

        np.clip(accurate_polygon_xy[:, 0], a_min=0., a_max=im_w, out=accurate_polygon_xy[:, 0])
        np.clip(accurate_polygon_xy[:, 1], a_min=0., a_max=im_h, out=accurate_polygon_xy[:, 1])
        np.clip(accurate_polygon_xyn, a_min=0., a_max=1., out=accurate_polygon_xyn)

        # NOTE: We are not clipping the quadrilateral to the image size, because we actively want it to be larger
        # than the polygon. It allows cropped QRs to be fully covered by the quadrilateral with only 4 points.

        expanded_quadrilateral_xy = np.array(_quadrilateral_fit.expanded_quadrilateral, dtype=np.float32)
        quadrilateral_xy = np.array(quadrilateral_xy, dtype=np.float32)

        expanded_quadrilateral_xyn = expanded_quadrilateral_xy/(im_w, im_h)
        quadrilateral_xyn = quadrilateral_xy/(im_w, im_h)

        assert len(boxes.conf) == 1, f'Expected confidence result to have length 1, got {len(result)}'
        confidence = float(boxes.conf[0])
        assert len(boxes.cls) == 1 and int(boxes.cls[0]) == 0, f'Expected class to be always [0], got {boxes.cls}'

        # Calculate center and width/height of the bounding box (post-clipping)
        cx, cy = float((bbox_xyxy[0] + bbox_xyxy[2])/2), float((bbox_xyxy[1] + bbox_xyxy[3])/2)
        bbox_w, bbox_h = float(bbox_xyxy[2] - bbox_xyxy[0]), float(bbox_xyxy[3] - bbox_xyxy[1])
        cxn, cyn, bbox_wn, bbox_hn = cx/im_w, cy/im_h, bbox_w/im_w, bbox_h/im_h

        detections.append({
            CONFIDENCE: confidence,

            BBOX_XYXY: bbox_xyxy,
            BBOX_XYXYN: bbox_xyxyn,
            CXCY: (cx, cy), CXCYN: (cxn, cyn),
            WH: (bbox_w, bbox_h), WHN: (bbox_wn, bbox_hn),

            POLYGON_XY: accurate_polygon_xy,
            POLYGON_XYN: accurate_polygon_xyn,
            QUAD_XY: quadrilateral_xy,
            QUAD_XYN: quadrilateral_xyn,
            PADDED_QUAD_XY: expanded_quadrilateral_xy,
            PADDED_QUAD_XYN: expanded_quadrilateral_xyn,

            IMAGE_SHAPE: (im_h, im_w),
        })

    crop_qr(image=image, detection=detections[0], crop_key=PADDED_QUAD_XYN)

    return detections

def _prepare_input(source: str | np.ndarray | 'PIL.Image' | 'torch.Tensor', is_bgr: bool = False) ->\
        str | np.ndarray|'PIL.Image'|'torch.Tensor':
    """
    Adjust the source if needed to be more flexible with the input. For example, it transforms
    grayscale images to RGB images. Or np.float (0.-1.) to np.uint8 (0-255)
    :param source: str|np.ndarray|PIL.Image|torch.Tensor. The source to get the expected channel order
                    from. str can be a path or url to an image. Images must be in RGB or BGR format. Can be
                    also 'screen' to take a screenshot
    :return: str|np.ndarray|PIL.Image|torch.Tensor. The adjusted source.
    """

    if isinstance(source, np.ndarray):
        if len(source.shape) == 2:
            # If it is grayscale, transform it to RGB by repeating the channel 3 times.
            source = np.repeat(source[:, :, np.newaxis], 3, axis=2)
            is_bgr = True
        assert len(source.shape) == 3, f'Expected image to have 3 dimensions (H, W, RGB). Got {source.shape}.'
        if source.shape[2] == 4:
            # If it is RGBA, transform it to RGB by removing the alpha channel.
            source = source[:, :, :3]
        assert source.shape[2] == 3, f'Expected image to have 3 or 4 channels (RGB[A]).' \
                                     f' Got {source.shape[2]}. (Alpha channel is always ignored)'
        if source.dtype in (np.float32, np.float64):
            # If it is float, transform it to uint8.
            assert np.min(source) >= 0. and np.max(source) <= 1., f"Expected image to be in range [0., 1.] if " \
                                                                  f"passed as float. Got [{np.min(source)}, " \
                                                                    f"{np.max(source)}]."
            source = (source * 255).astype(np.uint8)
        assert source.dtype == np.uint8, f'Expected image to be of type np.uint8. Got {source.dtype}.'
        if not is_bgr:
            # YoloV8 expects BGR images, so if it is RGB, transform it to BGR.
            source = source[:, :, ::-1]
    # For PIL.Image
    elif type(source).__name__.endswith('ImageFile'):
        # Cast to numpy array to make things easier.
        source = _prepare_input(source=np.array(source, dtype=np.uint8), is_bgr=is_bgr)
    # For torch.Tensor
    elif type(source).__name__ == 'Tensor':
        assert len(source.shape) == 4, f"Expected tensor to have 4 dimensions (B, C, H, W). Got {source.shape}."
        assert source.shape[1] == 3, f"Expected tensor to have 3 channels, with shape: (B, RGB, H, W). " \
                                     f"Got {source.shape}."
        assert str(source.dtype) == 'torch.float32', f"Expected tensor to be of type float32. Got {source.dtype}."
        assert source.min().item() >= 0.0 and source.max().item() <= 1.0, f"Expected tensor to be in range " \
                                                                          f"[0.0, 1.0]. Got" \
                                                                          f" [{source.min().item()}," \
                                                                          f" {source.max().item()}]."
        if is_bgr:
            # YoloV8 expects RGB images for torch.Tensors, so if it is BGR, transform it to RGB.
            source = source[:, [2, 1, 0], :, :]
    elif isinstance(source, str):
        # If it is a path, an url or 'screen', it is auto-managed
        pass
    else:
        raise TypeError(f"Expected source to be one of the following types: "
                        f"str|np.ndarray|PIL.Image|torch.Tensor. Got {type(source)}.")
    return source

def crop_qr(image: np.ndarray, detection: dict[str, np.ndarray|float|tuple[float|int, float|int]],
            crop_key: str = BBOX_XYXY) -> tuple[np.ndarray, dict[str, np.ndarray|float|tuple[float|int, float|int]]]:
    """
    Crop the QR code from the image.
    :param image: np.ndarray. The image to crop the QR code from.
    :param detection: dict[str, np.ndarray|float|tuple[float|int, float|int]]. The detection of the QR code returned
                      by QRDetector.detect(). Take into account that this method will return a tuple of detections,
                        and this function only expects one of them.
    :param crop_key: str. The key of the detection to use to crop the QR code. It can be one of the following:
                     'bbox_xyxy', 'bbox_xyn', 'quad_xy', 'quad_xyn', 'padded_quad_xy', 'padded_quad_xyn',
                     'polygon_xy' or 'polygon_xyn'.
    :return: tuple[np.ndarray, dict[str, np.ndarray|float|tuple[float|int, float|int]]]. The cropped QR code, and the
                    updated detection to fit the cropped image.
    """
    if crop_key in (BBOX_XYXYN, QUAD_XYN, PADDED_QUAD_XYN, POLYGON_XYN):
        # If it is normalized, transform it to absolute coordinates.
        crop_key = crop_key[:-1]

    # Get the surrounding box of the QR code
    # If it is the bbox, that's actually the box itself
    if crop_key == BBOX_XYXY:
        x1, y1, x2, y2 = detection[crop_key]
    # Otherwise, calculate the limits of the polygon
    else:
        (x1, y1), (x2, y2) = detection[crop_key].min(axis=0), detection[crop_key].max(axis=0)
    x1, y1, x2, y2 = floor(x1), floor(y1), ceil(x2), ceil(y2)

    h, w = image.shape[:2]
    # Apply pad if needed
    left_pad, top_pad = max(0, -x1), max(0, -y1)
    right_pad, bottom_pad = max(0, x2 - w), max(0, y2 - h)

    if any(pad > 0 for pad in (left_pad, top_pad, right_pad, bottom_pad)):
        white = 255 if image.dtype == np.uint8 else 1.
        assert np.max(image) <= white, f"Expected image to be in range [0, 255] if passed as np.uint8, or [0., 1.] " \
                                          f"if passed as float. Got [{np.min(image)}, {np.max(image)}]."
        # Apply pad if needed
        if len(image.shape) == 3:
            image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                            constant_values=white)
        else:
            assert len(image.shape) == 2, f"Expected image to have 2 or 3 dimensions. Got {len(image.shape)}."
            image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                            constant_values=white)
        h, w = image.shape[:2]
        # Update the coordinates
        x1, y1, x2, y2 = x1 + left_pad, y1 + top_pad, x2 + left_pad, y2 + top_pad
        assert x1 >= 0. and y1 >= 0. and x2 <= w and y2 <= h, f"Padded coordinates are out of bounds range."
        image = image[y1:y2, x1:x2]
        x1, y1 = x1 - left_pad, y1 - top_pad
    else:
        image = image[y1:y2, x1:x2].copy()

    # To avoid castings
    x1, y1 = np.float32(x1), np.float32(y1)
    # Update the detection to fit the cropped image
    h, w = image.shape[:2]
    pre_detection = detection.copy()
    detection = detection.copy()
    # Recalculate everything, (taking into account the padding)
    detection.update({
     BBOX_XYXY: np.array([0., 0., w, h], dtype=np.float32),
     CXCY: (w / 2., h / 2.),
     WH: (w, h),
     POLYGON_XY: detection[POLYGON_XY] - (x1, y1),
     QUAD_XY: detection[QUAD_XY] - (x1, y1),
     PADDED_QUAD_XY: detection[PADDED_QUAD_XY] - (x1, y1),
     IMAGE_SHAPE: (h, w),
    })
    # To avoid castings
    h, w = np.float32(h), np.float32(w)
    detection.update({
        BBOX_XYXYN: np.array((0., 0., 1., 1.), dtype=np.float32),
        CXCYN: (0.5, 0.5),
        WHN: (1., 1.),
        POLYGON_XYN: detection[POLYGON_XY] / (w, h),
        QUAD_XYN: detection[QUAD_XY] / (w, h),
        PADDED_QUAD_XYN: detection[PADDED_QUAD_XY] / (w, h),
    })

    return image, detection




def _plot_result(image: np.ndarray, detections: tuple[dict[str, np.ndarray|float|tuple[float, float]]]):
    """
    Plot the result of the detection on the image. Useful for debugging.
    :param image: np.ndarray. The image to plot the result on.
    :param detections: tuple[dict[str, np.ndarray|float|tuple[float, float]]]. The detections to plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot results. Install it with: pip install matplotlib")

    for detection in detections:
        # Plot confidence on the top of the bbox
        bbox_xyxy = detection[BBOX_XYXY]
        confidence = detection[CONFIDENCE]

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        ax.text(bbox_xyxy[0], bbox_xyxy[1] - 2, f'{confidence:.2f}', fontsize=10, color='yellow')

        # Plot the quadrilateral, tight quadrilateral and polygon
        quadrilateral_xy = detection[QUAD_XY]
        polygon_xy = detection[POLYGON_XY]
        # Repeat the first point to close the polygon
        polygon_xy = np.vstack((polygon_xy, polygon_xy[0]))
        quadrilateral_xy = np.vstack((quadrilateral_xy, quadrilateral_xy[0]))
        ax.plot(quadrilateral_xy[:, 0], quadrilateral_xy[:, 1], color='red', linestyle='-', linewidth=1, label='Quadrilateral')
        ax.plot(polygon_xy[:, 0], polygon_xy[:, 1], color='green', alpha=0.5, linestyle='-', marker='o', linewidth=1,
                label='Polygon')

        # Plot the bbox
        ax.add_patch(plt.Rectangle((bbox_xyxy[0], bbox_xyxy[1]), bbox_xyxy[2] - bbox_xyxy[0],
                                   bbox_xyxy[3] - bbox_xyxy[1], fill=False, color='blue', linewidth=1,
                                   linestyle='--', label='Bbox', alpha=0.6))

        plt.axis('off')
        plt.legend()
        plt.show()
        plt.close()