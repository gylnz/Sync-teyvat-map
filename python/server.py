import time

import cv2
import numpy as np
import asyncio
import websockets
import json
import math
import sys
import logging
import os
import cProfile

from windowcapture import WindowCapture
from typing import List

# from websockets import WebSocketServer

processing = []
wincap = WindowCapture('xCalculator')


class MinMatches:
    def __init__(self, min_deg: float, descriptor_match_i: cv2.DMatch = None, descriptor_match_j: cv2.DMatch = None):
        self.min_deg = min_deg
        self.descriptor_match_i = descriptor_match_i
        self.descriptor_match_j = descriptor_match_j


class Dimension:
    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.x = x
        self.y = y


class ImgDescriptor:
    def __init__(self, keypoint: cv2.KeyPoint, descriptor: cv2.UMat):
        self.keypoint = keypoint
        self.descriptor = descriptor


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


preview_name = "Preview"
log = setup_custom_logger('app')

dimensions = {
    2: Dimension("map", 7002, 1126),
    7: Dimension("enkanomiya", 1861, 1783),
    9: Dimension("sougan", 2027, 1958)
}
descriptors = {}
clientDimension = '2'
resources_path = '../'


def imshow(name: str, image):
    cv2.imshow(name, image)
    cv2.waitKey(1)


def txtshow(text: str):
    text_image = np.zeros((50, 220, 3), np.uint8)
    cv2.putText(text_image, text,
                (10, 30),  # bottomLeft
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                0.4,  # font scale
                (255, 255, 255),  # color
                1,  # thickness
                2)  # line style
    cv2.imshow("Message", text_image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def get_descriptors(name: str) -> ImgDescriptor:
    if name in descriptors:
        return descriptors[name]

    # Map Image Keypoints and Descriptors
    map_img_keypoints_file = open(f"{resources_path}data\\{name}ImgKeyPoints.dat")
    map_img_keypoints = json.load(map_img_keypoints_file)
    map_img_keypoints_file.close()

    for i, p in enumerate(map_img_keypoints):
        map_img_keypoints[i] = cv2.KeyPoint(x=p["pt"]["x"], y=p["pt"]["y"],
                                            size=p["size"], angle=p["angle"],
                                            response=p["response"], octave=p["octave"], class_id=p["class_id"])

    map_img_descriptors_file = open(f"{resources_path}data\\{name}ImgDescriptors.dat")
    map_img_descriptors = json.load(map_img_descriptors_file)
    map_img_descriptors_file.close()
    map_img_descriptors = np.array(map_img_descriptors, dtype=np.uint8)
    map_img_descriptors = cv2.UMat(map_img_descriptors)

    descriptors[name] = ImgDescriptor(map_img_keypoints, map_img_descriptors)
    return descriptors[name]


def tan_keypoint(keypoint_i: cv2.KeyPoint, keypoint_j: cv2.KeyPoint):
    map_rad_y = (keypoint_i.pt[1] -
                 keypoint_j.pt[1])
    map_rad_x = (keypoint_i.pt[0] -
                 keypoint_j.pt[0])

    return math.atan2(map_rad_y, map_rad_x)


async def find_map(dimension: Dimension):
    # Screenshot
    # os.system(f"screenshot {resources_path}img\\t.png")
    # target_img_ori = cv2.imread(f"{resources_path}img\\t.png", cv2.IMREAD_GRAYSCALE)

    target_img_ori = wincap.get_screenshot()
    target_img_ori = cv2.cvtColor(target_img_ori, cv2.COLOR_BGR2GRAY)

    # Region of interest
    x = int(target_img_ori.shape[1] * 0.03125)
    y = int(target_img_ori.shape[0] * 17 / 1080)
    w = int(target_img_ori.shape[0] * 215 / 1080)
    h = int(target_img_ori.shape[0] * 215 / 1080)

    region = (x, y, w, h)

    target_img = target_img_ori[y:y + h, x:x + w]

    # print(x,y, w, h)
    imshow(preview_name, target_img)
    # asyncio.create_task(imshow_async(preview_name, target_img))

    target_img = cv2.UMat(target_img)
    target_img = cv2.resize(target_img, (600, 600))

    # Target Image Keypoints and Descriptors
    akaze = cv2.AKAZE_create()
    target_img_keypoints = akaze.detect(target_img)
    target_img_descriptors = akaze.compute(target_img, target_img_keypoints)

    # output_image = cv2.drawKeypoints(target_img, target_img_keypoints, 0, (0, 255, 0),
    #                                  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # imshow(preview_name, output_image)

    descriptor = get_descriptors(dimension.name)
    map_img_keypoints = descriptor.keypoint
    map_img_descriptors = descriptor.descriptor

    # Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.match(target_img_descriptors[1], map_img_descriptors)

    if len(matches) == 0:
        return

    # Best Matches
    best_n = 40
    best_matches = sorted(matches, key=lambda x: x.distance)[:best_n]

    # imgB = cv2.imread(f"{resources_path}img\\map.png")
    # matched_image = cv2.drawMatches(target_img, target_img_keypoints, imgB, map_img_keypoints, best_matches, None, flags=2)
    # matched_image = cv2.resize(matched_image, (1920, 1080))
    # cv2.imshow("Match", matched_image)
    # cv2.waitKey(0)

    result_matches = []
    for i in range(len(best_matches) - 1):
        min_matches = MinMatches(360, best_matches[i])
        for j in range(i + 1, len(best_matches)):
            if i == j:
                continue

            map_rad = tan_keypoint(map_img_keypoints[best_matches[i].trainIdx],
                                   map_img_keypoints[best_matches[j].trainIdx])

            target_rad = tan_keypoint(target_img_keypoints[best_matches[i].queryIdx],
                                      target_img_keypoints[best_matches[j].queryIdx])

            map_deg = (map_rad * (180 / math.pi)) % 360
            target_deg = (target_rad * (180 / math.pi)) % 360
            map_deg = map_deg if map_deg >= 0 else 360 + map_deg
            target_deg = target_deg if target_deg >= 0 else 360 + target_deg
            diff_deg = abs(map_deg - target_deg)
            if math.isnan(diff_deg):
                continue
            if diff_deg < min_matches.min_deg:
                min_matches.min_deg = diff_deg
                min_matches.descriptor_match_j = best_matches[j]

        if min_matches.min_deg > 0.1 or min_matches.descriptor_match_i.distance > 50:
            continue
        result_matches.append(min_matches.descriptor_match_i)

        if min_matches.descriptor_match_j is None or min_matches.descriptor_match_j.distance > 50:
            continue
        result_matches.append(min_matches.descriptor_match_j)

    if len(result_matches) < 2:
        return
    result_matches = result_matches[:2]

    map_key_x = map_img_keypoints[result_matches[0].trainIdx].pt[0] - \
                map_img_keypoints[result_matches[1].trainIdx].pt[0]
    map_key_y = map_img_keypoints[result_matches[0].trainIdx].pt[1] - \
                map_img_keypoints[result_matches[1].trainIdx].pt[1]

    target_key_x = target_img_keypoints[result_matches[0].queryIdx].pt[0] - \
                   target_img_keypoints[result_matches[1].queryIdx].pt[0]
    target_rad_y = target_img_keypoints[result_matches[0].queryIdx].pt[1] - \
                   target_img_keypoints[result_matches[1].queryIdx].pt[1]

    mag = (
            np.linalg.norm(np.array([map_key_x, map_key_y], dtype=float)) /
            np.linalg.norm(np.array([target_key_x, target_rad_y], dtype=float))
    )

    res = np.multiply(np.array([
        300 - target_img_keypoints[result_matches[0].queryIdx].pt[0],
        300 - target_img_keypoints[result_matches[0].queryIdx].pt[1]
    ]), mag)

    res = np.add(res, np.array([
        map_img_keypoints[result_matches[0].trainIdx].pt[0],
        map_img_keypoints[result_matches[0].trainIdx].pt[1]
    ]))

    x, y = res

    msg = f"center={round(y - dimension.y, 2)},{round(x - dimension.x, 2)}"
    return msg


def find_map_callback(websocket: websockets.WebSocketServerProtocol, f, processing: List[bool], start: int):
    processing.clear()
    try:
        center = f.result()
        elapsed = time.perf_counter_ns() - start
        msg = f"[{center}] t:{elapsed / (1000 * 1000 * 1000)}s"
        log.info(msg)
        txtshow(msg)
        if center is not None:
            asyncio.create_task(websocket.send(center))
    except ValueError as e:
        log.error(e)


async def on_connect(websocket: websockets.WebSocketServerProtocol, path: str):
    log.info("connected")
    async for message in websocket:
        await on_message(websocket, path, message)


async def on_disconnect(websocket, path):
    log.info(f"Disconnected: {websocket}")


async def on_error(websocket, path, error):
    log.info(f"Error: {error}")


async def on_message(websocket, path, message):
    if len(processing) > 0:
        log.info(f"SKIP: {len(processing)}")
        return

    processing.append(True)
    key = int(message)
    dimension = dimensions.get(key, None)
    log.info(json.dumps(vars(dimension)))
    if dimension is not None:
        start = time.perf_counter_ns()
        future = asyncio.ensure_future(find_map(dimension))
        future.add_done_callback(lambda f: find_map_callback(websocket, f, processing, start))
    await websocket.send(message)


def run_find_map():
    dimension = dimensions[2]
    start = time.perf_counter_ns()
    result = asyncio.run(find_map(dimension))
    # asyncio.get_event_loop().run_forever()


def run_web_server():
    log.info(f"current directory: {os.getcwd()}")
    port = 27900
    log.info(f"starting WS on port {port}")
    start_server = websockets.serve(on_connect, "localhost", port)

    cv2.startWindowThread()
    cv2.namedWindow(preview_name)
    cv2.namedWindow("Message")

    # logger = logging.getLogger('websockets')
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(logging.StreamHandler())

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    # run_find_map()
    run_web_server()
