import os
os.environ["ALWAYSAI_DBG_DISABLE_MODEL_VALIDATION"] = "1"
import edgeiq
from edgeiq.processing.object_detection.types import PostProcessParams, PreProcessParams
from typing import List
import numpy as np
import cv2
import time


def yolo_v5_pre_process(params: PreProcessParams) -> np.ndarray:
    input_img = params.image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, params.size, [0, 0, 0], 1, crop=False)
    return blob


def yolo_v5_pre_process_trt(params: PreProcessParams) -> np.ndarray:
    input_img = params.image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, params.size, [0, 0, 0], 1, crop=False)
    return blob.ravel()


def yolo_v5_post_process(params: PostProcessParams):
    edgeiq_boxes: List[edgeiq.BoundingBox] = []
    edgeiq_confidences: List[float] = []
    edgeiq_indexes: List[int] = []

    outputs: np.ndarray = params.results
    input_image: np.ndarray = params.image
    INPUT_WIDTH: int = params.model_input_size[0]
    INPUT_HEIGHT: int = params.model_input_size[1]
    CONFIDENCE_THRESHOLD: float = params.confidence_level
    NMS_THRESHOLD: float = params.overlap_threshold
    SCORE_THRESHOLD: float = params.confidence_level

    class_ids = []
    confidences = []
    boxes = []
    # Data for rows
    rows = outputs.shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resize factors
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    # Iterate through detections
    for r in range(rows):
        row = outputs[0][r]
        confidence = row[4]
        # Discard weak detections
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            # Add detection data if class score above threshold
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                # get box data
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    # Perform non maxium suppression to eliminate unecessary boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        edgeiq_boxes.append(edgeiq.BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3]))
        edgeiq_confidences.append(confidences[i])
        edgeiq_indexes.append(class_ids[i])
    return edgeiq_boxes, edgeiq_confidences, edgeiq_indexes


def yolo_v5_post_process_optimized(params: edgeiq.ObjectDetectionPostProcessParams):
    start = time.time()
    edgeiq_boxes: List[edgeiq.BoundingBox] = []
    edgeiq_confidences: List[float] = []
    edgeiq_indexes: List[int] = []

    outputs: np.ndarray = params.results
    input_image: np.ndarray = params.image
    INPUT_WIDTH: int = params.model_input_size[0]
    INPUT_HEIGHT: int = params.model_input_size[1]
    CONFIDENCE_THRESHOLD: float = params.confidence_level
    NMS_THRESHOLD: float = params.overlap_threshold
    SCORE_THRESHOLD: float = params.confidence_level

    class_ids = []
    confidences = []
    boxes = []
    image_height, image_width = input_image.shape[:2]
    # Resize factors
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # Extract relevant columns from the outputs
    confidences = outputs[0][:, 4]
    classes_scores = outputs[0][:, 5:]

    # Create masks for confidence and score thresholds
    confidence_mask = confidences >= CONFIDENCE_THRESHOLD
    score_mask = np.max(classes_scores, axis=1) > SCORE_THRESHOLD

    # Combine masks to filter detections
    mask = confidence_mask & score_mask

    # Filter detections based on the combined mask
    filtered_rows = outputs[0][mask]

    # Extract necessary information
    confidences = filtered_rows[:, 4]
    class_ids = np.argmax(filtered_rows[:, 5:], axis=1)
    boxes = np.array([
        [
            int((row[0] - row[2] / 2) * x_factor),
            int((row[1] - row[3] / 2) * y_factor),
            int(row[2] * x_factor),
            int(row[3] * y_factor)
        ] for row in filtered_rows
    ])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Perform non maxium suppression to eliminate unecessary boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        edgeiq_boxes.append(edgeiq.BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3]))
        edgeiq_confidences.append(confidences[i])
        edgeiq_indexes.append(class_ids[i])
    print(f"Time to post process: {time.time() - start}")
    return edgeiq_boxes, edgeiq_confidences, edgeiq_indexes
