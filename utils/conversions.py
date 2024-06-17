
def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters


def pairs_to_points(pairs):
    """
    将成对的值转换为点。
    参数:
    pairs (list): 成对的值列表 [x1, y1, x2, y2, ...]
    返回:
    list: 点的列表 [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ...]
    """
    if len(pairs) % 2 != 0:
        raise ValueError("输入的列表长度必须为偶数")
    
    points = [[pairs[i], pairs[i+1]] for i in range(0, len(pairs), 2)]
    return points

def points_to_pairs(points):
    """
    将点转换回成对的值。
    参数:
    points (list): 点的列表 [[x1, y1], [x2, y2], ...]
    返回:
    list: 成对的值列表 [x1, y1, x2, y2, ...]
    """
    pairs = []
    for point in points:
        pairs.extend(point)
    return pairs
