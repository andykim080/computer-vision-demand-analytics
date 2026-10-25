"""Optional extra person views for difficult fixed-camera recordings."""
from pathlib import Path


def class_confidence_indices(rows, names, thresholds):
    """Optional class-specific detector cutoffs; no geometry or motion rules."""
    return [i for i, row in enumerate(rows)
            if row[4] >= (thresholds or {}).get(names[int(row[5])], 0)]


def person_keep_indices(rows, settings):
    """Optional camera-specific duplicate suppression; never uses motion or identity.

    A short box sharing a full body's top/center is often an extra torso/head
    detection. Offset/normally sized bodies are retained by the nested rule.
    Ordinary NMS is separately optional because it can merge occluded people.
    """
    import numpy as np
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    if not settings or not len(rows):
        return list(range(len(rows)))
    widths = np.maximum(0, rows[:, 2] - rows[:, 0])
    heights = np.maximum(0, rows[:, 3] - rows[:, 1])
    areas = widths * heights
    keep = np.ones(len(rows), dtype=bool)
    nested = settings.get('nested_height_ratio', 0)
    if nested:
        for outer in np.argsort(-areas):
            if not keep[outer] or areas[outer] <= 0:
                continue
            for inner in range(len(rows)):
                if inner == outer or not keep[inner] or areas[inner] <= 0:
                    continue
                if heights[inner] >= nested * heights[outer]:
                    continue
                overlap = np.maximum(0, np.minimum(rows[inner, 2:4], rows[outer, 2:4]) - np.maximum(rows[inner, :2], rows[outer, :2])).prod()
                aligned_top = abs(rows[inner, 1] - rows[outer, 1]) <= .15 * heights[outer]
                aligned_center = abs((rows[inner, 0]+rows[inner, 2]) - (rows[outer, 0]+rows[outer, 2])) / 2 <= .25 * widths[outer]
                if overlap / areas[inner] >= .9 and aligned_top and aligned_center:
                    keep[inner] = False
    indices = np.flatnonzero(keep)
    threshold = settings.get('iou', 1)
    if threshold < 1:
        retained = []
        for i in indices[np.argsort(-rows[indices, 4], kind='stable')]:
            if retained:
                intersection = np.maximum(0, np.minimum(rows[retained, 2:4], rows[i, 2:4]) - np.maximum(rows[retained, :2], rows[i, :2])).prod(axis=1)
                iou = intersection / np.maximum(areas[retained]+areas[i]-intersection, 1e-9)
                if np.any(iou > threshold):
                    continue
            retained.append(int(i))
        indices = np.array(sorted(retained), dtype=int)
    return indices.tolist()


def merge_people(rows, extra_rows, confidence, iou=0.5):
    """Fuse person-only passes without suppressing a vehicle or its rider."""
    import cv2
    import numpy as np
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    people = np.concatenate((rows[rows[:, 5] == 0],
                             np.asarray(extra_rows, dtype=np.float32).reshape(-1, 6)))
    if len(people):
        rectangles = people[:, :4].copy()
        rectangles[:, 2:] -= rectangles[:, :2]
        keep = cv2.dnn.NMSBoxes(rectangles.tolist(), people[:, 4].tolist(), confidence, iou)
        people = people[np.asarray(keep, dtype=int).reshape(-1)]
    return np.concatenate((rows[rows[:, 5] != 0], people))


class PersonViews:
    """Keep full-frame vehicles; add alternate-scale and optional tiled people."""
    def __init__(self, settings, root):
        from ultralytics import YOLO
        self.settings = settings
        checkpoint = Path(settings['model'])
        if not checkpoint.is_absolute():
            checkpoint = Path(root) / checkpoint
        self.model = YOLO(str(checkpoint))
        if self.model.names.get(0) != 'person':
            raise ValueError('Extra person model must use COCO class 0 for person')

    def predict(self, base, model, image, device, confidence, iou):
        import cv2
        import torch
        if model.names.get(0) != 'person':
            raise ValueError('Extra person views require a COCO person class at index 0')
        extra = []
        for spec in self.settings['passes']:
            flip = spec.get('flip', False)
            work = cv2.flip(image, 1) if flip else image
            result = self.model.predict(work, classes=[0], conf=confidence, iou=iou,
                                        imgsz=spec['imgsz'], device=device, verbose=False)[0]
            rows = result.boxes.data.cpu().numpy().copy()
            if flip:
                rows[:, [0, 2]] = image.shape[1] - rows[:, [2, 0]]
            extra.extend(rows.tolist())
        if self.settings.get('tiles', False):
            height, width = image.shape[:2]
            for x, y in ((0, 0), (int(width*.4), 0),
                         (0, int(height*.4)), (int(width*.4), int(height*.4))):
                crop = image[y:min(height, y+int(height*.6)), x:min(width, x+int(width*.6))]
                result = model.predict(crop, classes=[0], conf=confidence, iou=iou,
                                       imgsz=self.settings.get('tile_imgsz', 960), device=device, verbose=False)[0]
                rows = result.boxes.data.cpu().numpy().copy()
                rows[:, [0, 2]] += x
                rows[:, [1, 3]] += y
                extra.extend(rows.tolist())
        rows = merge_people(base.boxes.data.cpu().numpy(), extra, confidence,
                            self.settings.get('merge_iou', .5))
        base.update(boxes=torch.as_tensor(rows))
        return base
