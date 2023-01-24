import getopt
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

from imgfeature import init_feature, filter_matches, draw_inliers
from createbin import load_bin_file
from windowcapture import WindowCapture
from typing import List

# from websockets import WebSocketServer

processing = []
wincap = WindowCapture(None)
detector = None
matcher = None
static_img = None


def Object(**kwargs):
    return type("Object", (), kwargs)


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


class ImgFeatures:
    def __init__(self, keypoints: List[cv2.KeyPoint], descriptors: cv2.UMat):
        self.keypoints = keypoints
        self.descriptors = descriptors


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
img_features = {}
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


def get_img_features(name: str) -> ImgFeatures:
    if name in img_features:
        return img_features[name]

    keypoints, descriptors = load_bin_file(name)

    img_features[name] = ImgFeatures(keypoints, descriptors)
    return img_features[name]


def tan_keypoint(keypoint_i: cv2.KeyPoint, keypoint_j: cv2.KeyPoint):
    map_rad_y = (keypoint_i.pt[1] -
                 keypoint_j.pt[1])
    map_rad_x = (keypoint_i.pt[0] -
                 keypoint_j.pt[0])

    return math.atan2(map_rad_y, map_rad_x)


async def find_map(dimension: Dimension):
    if static_img is None:
        target_img_ori = wincap.get_screenshot()
        target_img_ori = cv2.cvtColor(target_img_ori, cv2.COLOR_BGR2GRAY)
    else:
        target_img_ori = static_img

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
    # target_img = cv2.resize(target_img, (600, 600))

    # Target Image Keypoints and Descriptors
    target_keypoints, target_descriptors = detector.detectAndCompute(target_img, None)

    # output_image = cv2.drawKeypoints(target_img, target_keypoints, 0, (0, 255, 0),
    #                                  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # imshow(preview_name, output_image)

    descriptor = get_img_features(dimension.name)
    map_keypoints = descriptor.keypoints
    map_descriptors = descriptor.descriptors

    log.info('matching...')
    inliers = 0
    matched = 0
    raw_matches = matcher.knnMatch(target_descriptors, trainDescriptors=map_descriptors, k=2)
    p1, p2, kp_pairs = filter_matches(target_keypoints, map_keypoints, raw_matches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        inliers = np.sum(status)
        matched = len(status)
        log.info('%d / %d  inliers/matched' % (inliers, matched))
    else:
        H, status = None, None
        log.info('%d matches found, not enough for homography estimation' % len(p1))
        if len(p1) > 0:
            vis = draw_inliers(target_img.get(), kp_pairs, status, True)
            imshow(preview_name, vis)
        return None, inliers, matched

    # map_img = cv2.imread(f"F:/data/p/Sync-teyvat-map/img/map.png", cv2.IMREAD_GRAYSCALE)
    # explore_match('find_obj', target_img.get(), map_img, kp_pairs, status, H)
    vis = draw_inliers(target_img.get(), kp_pairs, status)
    imshow(preview_name, vis)

    # imgB = cv2.imread(f"{resources_path}img\\map.png")
    # matched_image = cv2.drawMatches(target_img, target_keypoints, imgB, map_keypoints, result_matches, None, flags=4)
    # # matched_image = cv2.resize(matched_image, (1920, 1080))
    # plt.imshow(cv2.cvtColor(matched_image.get(), cv2.COLOR_BGR2RGB))
    # plt.show()

    h1, w1 = target_img.get().shape[:2]
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
    px = [p[0] for p in corners]
    py = [p[1] for p in corners]
    x, y = (sum(px) / len(corners), sum(py) / len(corners))

    msg = f"center={round(y - dimension.y, 2)},{round(x - dimension.x, 2)}"
    return msg, inliers, matched


def find_map_callback(websocket: websockets.WebSocketServerProtocol, f, processing: List[bool], start: int):
    processing.clear()
    try:
        center, inliers, matched = f.result()
        center_msg = center
        if center_msg is not None:
            center_msg = center_msg.split('=')[1]
        elapsed = time.perf_counter_ns() - start
        msg = f"{center_msg}|{inliers}/{matched}|{round(elapsed / (1000 * 1000 * 1000), 3)}s"
        log.info(msg)
        txtshow(msg)
        if center is not None and websocket is not None:
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
    global static_img
    if use_static_img:
        static_img = cv2.imread(f"D:/data/projects/data/5.png", cv2.IMREAD_GRAYSCALE)
    dimension = dimensions[2]
    start = time.perf_counter_ns()
    result = asyncio.run(find_map(dimension))
    find_map_callback(None, Object(result=lambda: result), [], start)
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
    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(argv, "ts", ["test", "static"])
    except getopt.GetoptError:
        print('server.py -t')
        sys.exit(2)

    test_mode = False
    use_static_img = False
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t -s')
            sys.exit()
        elif opt in ("-t", "--test"):
            test_mode = True
        elif opt in ("-s", "--static"):
            use_static_img = True

    detector, matcher = init_feature()

    if test_mode:
        run_find_map()
    else:
        run_web_server()
