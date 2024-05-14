import csv
import os
import time

import requests
import trimesh

import numpy as np


def download_file(url, output_path):
    res = requests.get(url)
    with open(output_path, "wb") as output:
        output.write(res.content)


class DataLoader:
    def __init__(self, server="https://easyvizar.wings.cs.wisc.edu", location_id="956d639d-69e0-4ff7-a58c-1e505e8e096a", cache_dir="cache"):
        self.server = server
        self.location_id = location_id

        self.cache_dir = cache_dir
        self.surfaces_dir = os.path.join(cache_dir, location_id, "surfaces")
        self.traces_dir = os.path.join(cache_dir, location_id, "traces")

    def fetch_surface(self, location_id, surface_id, ignore_cache=True):
        surfaces_dir = os.path.join(self.cache_dir, location_id, "surfaces")
        os.makedirs(surfaces_dir, exist_ok=True)

        file_name = "{}.ply".format(surface_id)
        file_path = os.path.join(surfaces_dir, file_name)

        if ignore_cache or not os.path.exists(file_path):
            url = "{}/locations/{}/surfaces/{}/surface.ply".format(self.server, location_id, surface_id)
            print("Downloading surface from {}".format(url))
            download_file(url, file_path)

    def load_cached_surfaces(self, location_id):
        """
        Load all cached surfaces from a given location.

        Returns trimesh mesh containing all of the surfaces.
        """
        surfaces_dir = os.path.join(self.cache_dir, location_id, "surfaces")
        
        surfaces = []
        for fname in os.listdir(surfaces_dir):
            path = os.path.join(surfaces_dir, fname)
            mesh = trimesh.load(path)
            surfaces.append(mesh)

        return trimesh.util.concatenate(surfaces)

    def load_surfaces(self):
        """
        Load surfaces as trimesh meshes from server or cached files.

        Returns:
            [N] list of trimesh objects
            [N] list of string UUIDs which consistently identify surfaces across updates
        """
        os.makedirs(self.surfaces_dir, exist_ok=True)

        surfaces = []
        surface_ids = []

        url = "{}/locations/{}/surfaces".format(self.server, self.location_id)
        res = requests.get(url)
        for item in res.json():
            file_name = "{}.ply".format(item['id'])
            file_path = os.path.join(self.surfaces_dir, file_name)
            if not os.path.exists(file_path):
                # Avoid triggering server rate limit.
                time.sleep(0.2)

                url = "{}/locations/{}/surfaces/{}/surface.ply".format(self.server, self.location_id, item['id'])
                print("Downloading surface from {}".format(url))
                download_file(url, file_path)

            mesh = trimesh.load(file_path)
            if isinstance(mesh, trimesh.Trimesh):
                surfaces.append(mesh)
                surface_ids.append(item['id'])

        return surfaces, surface_ids

    def load_traces(self):
        """
        Load user position history traces from server or cached files.

        Returns:
            [N] list of traces, each trace being a tuple of two numpy arrays
                (M, 1) timestamps in seconds
                (M, 3) points (x, y, z)
        """
        os.makedirs(self.traces_dir, exist_ok=True)

        traces = []

        url = "{}/locations/{}/check-ins".format(self.server, self.location_id)
        res = requests.get(url)
        for item in res.json():
            file_name = "pose-changes-{}.csv".format(item['id'])
            file_path = os.path.join(self.traces_dir, file_name)
            if not os.path.exists(file_path):
                # Avoid triggering server rate limit.
                time.sleep(0.2)

                url = "{}/headsets/{}/tracking-sessions/{}/pose-changes.csv".format(self.server, item['headset_id'], item['id'])
                print("Downloading trace from {}".format(url))
                res = requests.get(url)
                download_file(url, file_path)

            with open(file_path, "r") as source:
                times = []
                points = []

                reader = csv.DictReader(source)
                for line in reader:
                    times.append(float(line['time']))
                    points.append([
                        float(line['position.x']),
                        float(line['position.y']),
                        float(line['position.z'])
                    ])

                traces.append((np.array(times), np.array(points)))

        return traces
