import json
import os
import sys
import time
import traceback

from http import HTTPStatus

import numpy as np

import marshmallow
import quaternion
import requests
import trimesh
import websocket

from .dataloader import DataLoader
from .photo import Photo


CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
DISPLAY = os.environ.get("DISPLAY")

QUEUE_NAME = os.environ.get("QUEUE_NAME", "detection-3d")
NEXT_QUEUE_NAME = os.environ.get("NEXT_QUEUE_NAME", "done")


MARK_CLASSES = set([
    "chair"
])


def cylinder_contains_any(points, center, radius=1, height=1):
    if len(points) == 0:
        return False

    distances = np.linalg.norm(points[:, [0, 2]] - center[[0, 2]], axis=1)
    print("distances: ", distances)
    within_range = distances < radius
    above_bottom = points[:, 1] > center[1] - (height / 2)
    below_top = points[:, 1] < center[1] + height # look higher up than down

    return np.any(within_range & above_bottom & below_top)


def vertical_cylinder_transform(center):
    transform = np.zeros((4, 4))

    # Hard-coded 90-degree rotation about the X axis to make a vertical cylinder.
    transform[0, 0] = 1
    transform[1, 2] = -1
    transform[2, 1] = 1
    transform[3, 3] = 1

    # Add translation to the center point
    transform[0:3, 3] = center

    return transform


class MapperClient:
    def __init__(self, server):
        self.server = server
        self.loader = DataLoader(server=server, cache_dir=CACHE_DIR)

    def on_close(self, ws, status_code, message):
        print("Connection closed with message: {} ({})".format(message, status_code))

    def on_error(self, ws, error):
        print(error)
        traceback.print_tb(error.__traceback__)

    def on_message(self, ws, message):
        print(message)
        data = json.loads(message)

        if data['event'].startswith("surfaces:"):
            self.on_surface_changed(data)

        elif data['event'] == "photos:updated":
            self.on_photo_updated(data)

    def on_open(self, ws):
        print("Connected to {}".format(self.server))
        ws.send("hold 3600")
        ws.send("subscribe surfaces:created *")
        ws.send("subscribe surfaces:updated *")
        ws.send("subscribe photos:updated *")

    def find_objects_in_photo(self, photo):
        if not photo.is_situated():
            return
        location_id = str(photo.camera_location_id)

        if len(photo.annotations) == 0:
            return

        photo_file = photo.get_file("photo")
        if photo_file is None:
            return

        cache = self.loader.cache_contents(location_id)
        if cache.surfaces == 0:
            self.loader.fetch_surfaces(location_id)

        mesh = self.loader.load_cached_surfaces(location_id)
        print(mesh)

        # Mirror the mesh about the X axis to be consistent
        # with the right-handed convention in trimesh.
#        mesh.apply_scale([-1, 1, 1])

        resolution = [photo_file.height, photo_file.width]
        center = photo.camera_position.as_array()
        rot_mat = photo.camera_orientation.as_rotation_matrix()
        fx, fy, cx, cy = photo.camera.relative_parameters()

        directions = []
        sizes = []
        for annotation in photo.annotations:
            if annotation.label not in MARK_CLASSES:
                continue

            pos = annotation.boundary.center()
            direction = np.array([
                (pos[0] - cx) / fx,
                (cy - pos[1]) / fy,
                1
            ])

            # Multiply by the camera rotation matrix to produce
            # direction vector in world coordinate frame.
            direction = np.matmul(rot_mat, direction)
            directions.append(direction)

            sizes.append([
                annotation.boundary.width / fx,
                annotation.boundary.height / fy
            ])

        if len(directions) == 0:
            return

        origins = [np.array(center)] * len(directions)

        # Ray cast against the environment mesh.
        points, index_ray, index_tri = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        print(points)
        print(index_ray)
        if len(points) == 0:
            return

        features = self.loader.load_features(location_id)
        feature_points = np.array([f['position'] for f in features])

        scene = mesh.scene()

        # Axis at world coordinate system origin.
        world_axis = trimesh.creation.axis(origin_size=0.2)
        scene.add_geometry(world_axis)

        cam = np.eye(4)
        cam[0:3, 0:3] = rot_mat
        cam[0:3, 3] = center
        cam_axis = trimesh.creation.axis(origin_size=0.1, transform=cam, origin_color=[0, 0, 255, 255])
        scene.add_geometry(cam_axis)

        distances = np.linalg.norm(points - center, axis=1)
        sizes = distances[:, np.newaxis] * np.array(sizes)[index_ray, :]

        # Add a cylinder for each predicted object location.
        for i, point in enumerate(points):
            width, height = sizes[i, :]
            half_height = 0.5 * height
            radius = 0.5 * width
            color = [0, 255, 0, 96]

            # Check if any existing features are within this expanded cylinder.
            if not cylinder_contains_any(feature_points, point, width, height):
                name = photo.annotations[index_ray[i]].label
                marker_point = point + [0, half_height, 0]
                self.loader.create_feature(location_id, "object", name, marker_point)

                # Append to the list of features from the server so that we do
                # not create duplicate features even if the image has
                # overlapping bounding boxes for some reason.
                feature_points = np.vstack([feature_points, marker_point])

                color = [0, 255, 0, 192]

            obj_transform = vertical_cylinder_transform(point)
            marker = trimesh.creation.cylinder(radius=radius, height=height, transform=obj_transform, face_colors=[0, 255, 0, 192])
            scene.add_geometry(marker)

        if DISPLAY is not None:
            scene.show()

    def on_photo_updated(self, data):
        photo = Photo.Schema(unknown=marshmallow.EXCLUDE).load(data['current'])
        print(photo)

        if photo.queue_name != QUEUE_NAME:
            return

        self.find_objects_in_photo(photo)
        self.loader.set_photo_queue(photo.id, NEXT_QUEUE_NAME)

    def on_surface_changed(self, data):
        words = data['uri'].split('/')
        if len(words) < 5:
            return

        location_id = words[2]
        surface_id = words[4]

        cache = self.loader.cache_contents(location_id)
        if cache.surfaces == 0:
            self.loader.fetch_surfaces(location_id)
        else:
            self.loader.fetch_surface(location_id, surface_id)

    def run(self):
        if self.server.startswith("https"):
            ws_server = self.server.replace("https", "wss")
        else:
            ws_server = self.server.replace("http", "ws")

        wsapp = websocket.WebSocketApp(ws_server + "/ws",
                on_close=self.on_close, on_error=self.on_error,
                on_open=self.on_open, on_message=self.on_message,
                on_reconnect=self.on_open)
        while True:
            wsapp.run_forever(ping_interval=30, ping_timeout=15,
                    ping_payload="ping", reconnect=15)
            time.sleep(15)
