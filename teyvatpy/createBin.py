import cv2
import numpy as np
import os
import struct

from typing import List


keypoint_format = "dddddii"
homePath = os.path.dirname(__file__).split('teyvatpy')[0]
detector = cv2.AKAZE_create()


def save_bin_file(name: str):
    map_img = cv2.imread(f"{homePath}/img/{name}.png", cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = detector.detectAndCompute(map_img, None)
    write_keypoints(f"{homePath}/data/{name}ImgKeyPoints.bin", keypoints)
    np.save(f"{homePath}/data/{name}ImgDescriptors", descriptors)


def load_bin_file(name: str) -> (List[cv2.KeyPoint], np.ndarray):
    keypoints = read_keypoints(f"{homePath}/data/{name}ImgKeyPoints.bin")
    descriptors = np.load(f"{homePath}/data/{name}ImgDescriptors.npy")
    return keypoints, descriptors


def write_keypoints(filename: str, keypoints: List[cv2.KeyPoint]):
    with open(filename, "wb") as file:
        for keypoint in keypoints:
            data = struct.pack(keypoint_format, keypoint.pt[0], keypoint.pt[1],
                               keypoint.size, keypoint.angle,
                               keypoint.response, keypoint.octave, keypoint.class_id)
            file.write(data)


def read_keypoints(filename: str) -> List[cv2.KeyPoint]:
    with open(filename, "rb") as file:
        struct_size = struct.calcsize(keypoint_format)
        keypoints = []
        while True:
            bts = file.read(struct_size)
            if not bts:
                break
            x, y, size, angle, response, octave, class_id = struct.unpack(keypoint_format, bts)
            keypoints.append(cv2.KeyPoint(x=x, y=y, size=size,
                                          angle=angle, response=response,
                                          octave=octave, class_id=class_id))
        return keypoints


if __name__ == "__main__":
    maps = ["map", "enkanomiya", "sougan"]
    length = len(maps)
    for i, name in enumerate(maps):
        print(f"[{i+1}/{length}] create {name} data...")
        save_bin_file(name)
    print("done")
