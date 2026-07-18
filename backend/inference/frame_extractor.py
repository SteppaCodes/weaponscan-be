import cv2
from dataclasses import dataclass
from typing import Generator, Tuple
import numpy as np


@dataclass
class FrameInfo:
    """Metadata for an extracted frame."""
    frame: np.ndarray       # The actual pixel data
    index: int              # Frame number in the original video
    timestamp: float        # Seconds into the video
    is_last: bool           # True if this is the final frame


class FrameExtractor:
    """
    Extract frames from a video at a configurable interval.
    
    Memory-safe: yields one frame at a time. The video file
    is never loaded entirely into RAM regardless of size.
    
    Args:
        video_path: Path to the video file
        skip_frames: Process every Nth frame (1 = every frame)
    """

    def __init__(self, video_path: str, skip_frames: int = 3):
        self.video_path = video_path
        self.skip_frames = max(1, skip_frames)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.frames_to_process = self.total_frames // self.skip_frames

    def extract(self) -> Generator[FrameInfo, None, None]:
        """
        Yield frames one at a time.
        
        Only yields every Nth frame (controlled by skip_frames).
        Each frame is yielded with its metadata.
        """
        frame_index = 0

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    break

                if frame_index % self.skip_frames == 0:
                    timestamp = frame_index / self.fps if self.fps > 0 else 0
                    remaining = self.total_frames - frame_index

                    yield FrameInfo(
                        frame=frame,
                        index=frame_index,
                        timestamp=timestamp,
                        is_last=(remaining <= self.skip_frames),
                    )

                frame_index += 1

        finally:
            self.cap.release()

    def __del__(self): 
        """Ensure the video handle is released."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


            
