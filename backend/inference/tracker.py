from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Track:
    """A tracked detection across multiple frames."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple                # (x1, y1, x2, y2)
    frames_seen: int = 1
    frames_missing: int = 0



class DetectionTracker:
    """
    Track detections across frames for temporal stability.
    
    A detection is only "confirmed" after appearing in
    min_frames consecutive frames. This eliminates single-frame
    false positives.
    
    Args:
        min_frames: Frames before confirming a detection
        max_missing: Frames to keep a track alive without a match
        iou_threshold: Minimum IoU to match a detection to a track
    """
    
    def __init__(self, min_frames=3, max_missing=5, iou_threshold=0.3):
        self.min_frames = min_frames
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0

    def update(self, boxes, classes, confidences, class_names) -> List[Track]:
        """
        Update tracker with new frame detections.
        
        Returns only confirmed (stable) detections.
        """
        # Match detections to existing tracks
        matched_tracks = set()
        matched_dets = set()

        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_idx = None

            for i, (box, cls) in enumerate(zip(boxes, classes)):
                if i in matched_dets or int(cls) != track.class_id:
                    continue
                iou = self._iou(track.bbox, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx is not None:
                track.bbox = tuple(boxes[best_idx])
                track.confidence = float(confidences[best_idx])
                track.frames_seen += 1
                track.frames_missing = 0
                matched_tracks.add(tid)
                matched_dets.add(best_idx)
            else:
                track.frames_missing += 1
                if track.frames_missing > self.max_missing:
                    del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for i in range(len(boxes)):
            if i not in matched_dets:
                cls_id = int(classes[i])
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    class_id=cls_id,
                    class_name=class_names.get(cls_id, str(cls_id)),
                    confidence=float(confidences[i]),
                    bbox=tuple(boxes[i]),
                )
                self.next_id += 1
        
        # Return only confirmed tracks
        return [
            t for t in self.tracks.values()
            if t.frames_seen >= self.min_frames
        ]
    
    @staticmethod
    def _iou(box1, box2):
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
