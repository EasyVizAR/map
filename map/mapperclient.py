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


# Do not label objects which appear very small for some reason.  It could be
# due to visual occlusion or an error in the distance estimation.
MINIMUM_WIDTH = 0.2
MINIMUM_HEIGHT = 0.2


MARK_CLASSES = set([
    "chair"
])


# Skip contour projection for these classes because
# they are not expected to be part of the world mesh.
EXCLUDE_CONTOURS = set([
    "face",
    "person"
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
    def __init__(self, server, config):
        self.server = server
        self.config = config
        self.loader = DataLoader(server=server, cache_dir=CACHE_DIR)

        self.enable_contours = config.get("enable-contours", True)
        self.enable_features = config.get("enable-features", False)
        self.next_queue_name = config.get("next-queue-name", "done")
        self.queue_name = config.get("queue-name", "detection-3d")

    def on_close(self, ws, status_code, message):
        print("Connection closed with message: {} ({})".format(message, status_code))

    def on_error(self, ws, error):
        print(error)
        traceback.print_tb(error.__traceback__)

    def on_message(self, ws, message):
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

    def project_contour(self, photo, contour, mesh, distance=None):
        center = photo.camera_position.as_array()
        rot_mat = photo.camera_orientation.as_rotation_matrix()
        fx, fy, cx, cy = photo.camera.relative_parameters()

        print(contour.shape)

        # Contour is a list of x, y coordinates in pixel space.
        # Find the direction of a ray passing through each point.
        directions = np.ones((contour.shape[0], 3))
        directions[:, 0] = (contour[:, 0] - cx) / fx
        directions[:, 1] = (cy - contour[:, 1]) / fy

        # Multiply by the camera rotation matrix to produce
        # direction vector in world coordinate frame.
        directions = np.matmul(directions, rot_mat.T)

        if distance is None:
            # Ray cast against the environment mesh.
            origins = [np.array(center)] * len(directions)
            points, index_ray, index_tri = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        else:
            # Project the contour as if it were on a plane at a distance from
            # the camera.  This works better if the contour does not cleanly
            # match up with the mesh, but it will look funny if viewed from a
            # different direction.
            points = distance * directions + center

        return points

    def find_objects_in_photo(self, photo):
        if not photo.is_situated():
            return
        location_id = str(photo.camera_location_id)

        if len(photo.annotations) == 0:
            return

        photo_file = photo.get_file("photo")
        if photo_file is None:
            return

        mesh = self.loader.load_surfaces(location_id)
        print(mesh)

        # Mirror the mesh about the X axis to be consistent
        # with the right-handed convention in trimesh.
#        mesh.apply_scale([-1, 1, 1])

        resolution = [photo_file.height, photo_file.width]
        center = photo.camera_position.as_array()
        rot_mat = photo.camera_orientation.as_rotation_matrix()
        fx, fy, cx, cy = photo.camera.relative_parameters()

        scene = mesh.scene()

        directions = []
        sizes = []
        for i, annotation in enumerate(photo.annotations):
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
        if len(features) > 0:
            feature_points = np.array([f['position'] for f in features])
        else:
            feature_points = np.empty((0, 3))

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

            # Index into the original annotations array
            ai = index_ray[i]
            annotation = photo.annotations[ai]
            name = annotation.label

            if width < MINIMUM_WIDTH or height < MINIMUM_HEIGHT:
                print("Skipping object with small width and height ({}, {})".format(width, height))
                continue

            # Check if any existing features are within this expanded cylinder.
            if self.enable_features and name in MARK_CLASSES and not cylinder_contains_any(feature_points, point, width, height):
                marker_point = point + [0, half_height, 0]
                self.loader.create_feature(location_id, "object", name, marker_point)

                # Append to the list of features from the server so that we do
                # not create duplicate features even if the image has
                # overlapping bounding boxes for some reason.
                feature_points = np.vstack([feature_points, marker_point])

            obj_transform = vertical_cylinder_transform(point)
            marker = trimesh.creation.cylinder(radius=radius, height=height, transform=obj_transform, face_colors=[0, 255, 0, 128])
            scene.add_geometry(marker)

            if self.enable_contours and name not in EXCLUDE_CONTOURS and len(annotation.contour) > 0:
                pcontour = self.project_contour(photo, np.array(annotation.contour), mesh, distance=distances[i])
                self.loader.update_photo_annotation(annotation.id, projected_contour=pcontour.tolist())

                line = trimesh.path.entities.Line(list(range(len(pcontour))), color=[0, 0, 255, 255])
                path = trimesh.path.path.Path3D([line], np.array(pcontour))
                scene.add_geometry(path)

        if DISPLAY is not None:
            scene.show()

    def on_photo_updated(self, data):
        photo = Photo.Schema(unknown=marshmallow.EXCLUDE).load(data['current'])
        if photo.queue_name != self.queue_name:
            return

        self.find_objects_in_photo(photo)
        self.loader.set_photo_queue(photo.id, self.next_queue_name)

    def on_surface_changed(self, data):
        words = data['uri'].split('/')
        if len(words) < 5:
            return

        location_id = words[2]
        surface_id = words[4]

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
