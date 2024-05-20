import uuid

from dataclasses import field
from typing import Optional

import marshmallow
from marshmallow_dataclass import dataclass

import numpy as np
import quaternion


@dataclass
class Box:
    left: float = 0
    top: float = 0
    width: float = 0
    height: float = 0

    def center(self):
        return np.array([
            self.left + 0.5 * self.width,
            self.top + 0.5 * self.height
        ])


@dataclass
class Orientation:
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 0

    def as_array(self):
        return np.array([self.x, self.y, self.z, self.w])

    def as_quaternion(self):
        return np.quaternion(self.w, self.x, self.y, self.z)

    def as_rotation_matrix(self):
        return quaternion.as_rotation_matrix(self.as_quaternion())


@dataclass
class Position:
    x: float = 0
    y: float = 0
    z: float = 0

    def as_array(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Annotation:
    class Meta:
        unknown = marshmallow.EXCLUDE

    boundary: Box = field(default_factory=Box)
    confidence: float = 0
    label: str = ""
    sublabel: str = ""


@dataclass
class Camera:
    class Meta:
        unknown = marshmallow.EXCLUDE

    type: str = "color"
    width: int = 0
    height: int = 0
    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    def relative_parameters(self):
        return (self.fx / self.width,
                self.fy / self.height,
                self.cx / self.width,
                self.cy / self.height)


@dataclass
class File:
    class Meta:
        unknown = marshmallow.EXCLUDE

    content_type: str = "image/jpeg"
    height: int = 0
    width: int = 0
    name: str = "photo.jpg"
    purpose: str = "photo"


@dataclass
class Photo:
    class Meta:
        unknown = marshmallow.EXCLUDE

    id: int = 0
    priority: int = 0
    queue_name: str = "created"
    annotations: list[Annotation] = field(default_factory=list)
    files: list[File] = field(default_factory=list)
    camera: Optional[Camera] = None
    camera_location_id: Optional[uuid.UUID] = None
    camera_position: Optional[Position] = None
    camera_orientation: Optional[Orientation] = None

    def get_file(self, purpose):
        for f in self.files:
            if f.purpose == purpose:
                return f
        return None

    def is_situated(self):
        if self.camera is None or self.camera.width <= 0 or self.camera.height <= 0:
            return False
        if self.camera_location_id is None:
            return False
        if self.camera_position is None:
            return False
        if self.camera_orientation is None:
            return False
        return True
