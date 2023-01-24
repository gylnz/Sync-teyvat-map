import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6

DETECTOR = 'brisk'
MATCHER = ''


def init_feature():
    detector_name = DETECTOR
    matcher_name = MATCHER
    if detector_name == 'sift':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector_name == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif detector_name == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif detector_name == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif detector_name == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' == matcher_name:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def draw_inliers(img1, kp_pairs, status=None, failed=False):
    h1, w1 = img1.shape[:2]
    vis = np.zeros((h1, w1), np.uint8)
    vis[:h1, :w1] = img1
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = []
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    if failed:
        green = red
    for (x1, y1), inlier in zip(p1, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)

    return vis
