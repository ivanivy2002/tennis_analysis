import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def adjust_keypoints_to_lines(image, keypoints):
    # 转换为HSV图像
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义白色的HSV范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # 根据HSV范围创建一个白色区域的掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 寻找白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将关键点转换为numpy数组
    keypoints = np.array(keypoints)

    # 使用DBSCAN进行聚类，找出关键点与线条的靠近关系
    db = DBSCAN(eps=20, min_samples=2).fit(keypoints)
    labels = db.labels_

    # 将关键点移到最近的白色线上
    for i, point in enumerate(keypoints):
        label = labels[i]
        if label != -1:
            # 找到同一类的点
            same_cluster_points = keypoints[labels == label]
            # 计算这些点到白色区域的质心的距离
            distances = [cv2.pointPolygonTest(contour, (point[0], point[1]), True) for contour in contours]
            min_distance_index = np.argmin(np.abs(distances))
            closest_contour = contours[min_distance_index]
            # 将点移动到白色线上
            M = cv2.moments(closest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                keypoints[i] = (cx, cy)

    return keypoints

# 示例使用
# image = cv2.imread("path_to_image")
# keypoints = [(x1, y1), (x2, y2), ...]
# adjusted_keypoints = adjust_keypoints_to_lines(image, keypoints)
