import numpy as np
from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

from utils.bbox_utils import get_center_of_bbox

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # 加载YOLO模型

    def interpolate_ball_positions(self, ball_positions):
        # 提取球位置
        ball_positions = [x.get(1, []) for x in ball_positions]
        # 将列表转换为pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # 插值缺失值并向后填充
        df_ball_positions = df_ball_positions.interpolate().bfill()

        # 将DataFrame转换回列表
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        # 提取球位置
        ball_positions = [x.get(1, []) for x in ball_positions]
        # 将列表转换为pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0  # 初始化球击打列
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2  # 计算中间Y值
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()  # 滚动平均
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()  # 计算差分
        minimum_change_frames_for_hit = 25

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            # 检查位置变化
            negative_position_change = df_ball_positions.loc[i, 'delta_y'] > 0 and df_ball_positions.loc[i + 1, 'delta_y'] < 0
            positive_position_change = df_ball_positions.loc[i, 'delta_y'] < 0 and df_ball_positions.loc[i + 1, 'delta_y'] > 0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions.loc[i, 'delta_y'] > 0 and df_ball_positions.loc[change_frame, 'delta_y'] < 0
                    positive_position_change_following_frame = df_ball_positions.loc[i, 'delta_y'] < 0 and df_ball_positions.loc[change_frame, 'delta_y'] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
            
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]  # 使用YOLO模型预测帧

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame


    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # 绘制边界框
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                frame = self.draw_triangle(frame, bbox, (0, 255, 255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
