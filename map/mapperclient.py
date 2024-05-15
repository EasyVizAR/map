import json
import os

from http import HTTPStatus

import numpy as np

import quaternion
import requests
import trimesh
import websocket

from .dataloader import DataLoader


# Camera intrinsic parameters
W = 896
H = 594
FX = 700 / W
FY = 702 / H
CX = 454 / W
CY = 219 / H


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
        self.loader = DataLoader(server=server)

    def on_photo_updated(self, data):
        photo = data['current']

        location_id = photo.get("camera_location_id")
        if location_id is None:
            return

        photo_file = None
        for f in photo.get("files", []):
            if f['purpose'] == "photo":
                photo_file = f
                break
        if photo_file is None:
            return

        annotations = photo.get('annotations', [])
        if len(annotations) == 0:
            return

        # Need to do an extra query to get all of the photo information
        url = "{}/photos/{}".format(self.server, photo['id'])
        response = requests.get(url)
        if response.ok and response.status_code == HTTPStatus.OK:
            photo = response.json()
        else:
            return

        position = photo.get("camera_position")
        orientation = photo.get("camera_orientation")
        if None in (position, orientation):
            return

        cache = self.loader.cache_contents(location_id)
        if cache.surfaces == 0:
            self.loader.fetch_surfaces(location_id)

        mesh = self.loader.load_cached_surfaces(location_id)
        print(mesh)

        # Mirror the mesh about the X axis to be consistent
        # with the right-handed convention in trimesh.
#        mesh.apply_scale([-1, 1, 1])

        resolution = [photo_file['height'], photo_file['width']]
        center = [
            photo['camera_position']['x'],
            photo['camera_position']['y'],
            photo['camera_position']['z'],
        ]
        orientation = np.quaternion(
            photo['camera_orientation']['w'],
            photo['camera_orientation']['x'],
            photo['camera_orientation']['y'],
            photo['camera_orientation']['z']
        )

        rot_mat = quaternion.as_rotation_matrix(orientation)

        directions = []
        for annotation in annotations:
            bbox = annotation['boundary']
            x = bbox['left'] + 0.5 * bbox['width']
            y = bbox['top'] + 0.5 * bbox['height']

            direction = np.array([
                (x - CX) / FX,
                (CY - y) / FY,
                1
            ])

            # Multiply by the camera rotation matrix to produce
            # direction vector in world coordinate frame.
            direction = np.matmul(rot_mat, direction)
            directions.append(direction)

        origins = [np.array(center)] * len(directions)

        # Ray cast against the environment mesh.
        points, index_ray, index_tri = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        print(points)
        print(index_ray)

        scene = mesh.scene()

        # Axis at world coordinate system origin.
        world_axis = trimesh.creation.axis(origin_size=0.2)
        scene.add_geometry(world_axis)

        cam = np.eye(4)
        cam[0:3, 0:3] = rot_mat
        cam[0:3, 3] = center
        cam_axis = trimesh.creation.axis(origin_size=0.1, transform=cam, origin_color=[0, 0, 255, 255])
        scene.add_geometry(cam_axis)

        distances = np.linalg.norm(points - center, axis=0)
        print(distances)

        # Add a cylinder for each predicted object location.
        for i, point in enumerate(points):
            # It is possible not all rays hit something or that the solver changed
            # the order of the rays. This gives us the index into the original data.
            j = index_ray[i]

            height = annotations[j]['boundary']['height'] * distances[i] / FY
            width = annotations[j]['boundary']['width'] * distances[i] / FX

            obj_transform = vertical_cylinder_transform(point)
            marker = trimesh.creation.cylinder(radius=width/2, height=height, transform=obj_transform, face_colors=[0, 255, 0, 192])
            scene.add_geometry(marker)

        scene.show()

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

    def open_websocket(self):
        if self.server.startswith("https"):
            ws_server = self.server.replace("https", "wss")
        else:
            ws_server = self.server.replace("http", "ws")

        ws = websocket.WebSocket()
        ws.connect(ws_server + "/ws")
        print("Connected to {}".format(ws_server))

        ws.send("subscribe surfaces:created *")
        ws.send("subscribe surfaces:updated *")
        ws.send("subscribe photos:updated *")

        return ws

    def run(self):
        ws = self.open_websocket()
        while True:
            msg = ws.recv()
            data = json.loads(msg)

            if data['event'].startswith("surfaces:"):
                self.on_surface_changed(data)

            elif data['event'] == "photos:updated":
                self.on_photo_updated(data)
