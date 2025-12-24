def compute_weight_index(bbox, frame_area):
    """
    bbox: [x1, y1, x2, y2]
    frame_area: width * height
    """
    x1, y1, x2, y2 = bbox
    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    return area / frame_area if frame_area > 0 else 0.0
