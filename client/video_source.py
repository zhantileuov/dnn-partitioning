from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class LoopingVideoFrameSource:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.cap = None
        self.frame_id = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

    def read(self):
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame from video: {self.video_path}")
        frame_id = self.frame_id
        self.frame_id += 1
        return frame_id, frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
