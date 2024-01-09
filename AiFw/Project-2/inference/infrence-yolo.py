from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_path = './test/000000.jpg'

weight = './weight/tiger-x.pt'

img_bgr = cv2.imread(img_path)  # 读取图片


def run_to_yolo(img_bgr, weight):
    start_time = time.time()
    model = YOLO(weight)

    model.to(device)
    results = model(img_bgr)

    bbox_color = (150, 0, 0)  # 框的 BGR 颜色
    bbox_thickness = 2  # 框的线宽
    bbox_labelstr = {
    'font_size': 2,  # 字体大小
    'font_thickness': 4,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': -20,  # Y 方向，文字偏移距离，向下为正
    }
    kpt_color_map = {
    0: {'name': '1-1', 'color': [0, 0, 255], 'radius': 6},  # 左 前 脚
    1: {'name': '1-2', 'color': [255, 0, 0], 'radius': 6},  # 右 前 脚

    2: {'name': '2-1', 'color': [0, 255, 0], 'radius': 6},  # 左 后 脚
    3: {'name': '2-2', 'color': [0, 255, 0], 'radius': 6},  # 右 后 脚

    4: {'name': '3-1', 'color': [0, 255, 0], 'radius': 6},  # 左 前 腿
    5: {'name': '3-2', 'color': [0, 255, 0], 'radius': 6},  # 右 前 腿

    6: {'name': '4-1', 'color': [0, 255, 0], 'radius': 6},  # 左 后 腿
    7: {'name': '4-2', 'color': [0, 255, 0], 'radius': 6},  # 右 后 腿

    8: {'name': '5-1', 'color': [0, 255, 0], 'radius': 6},  # 左 耳
    9: {'name': '5-2', 'color': [0, 255, 0], 'radius': 6},  # 右 耳

    10: {'name': '6', 'color': [0, 255, 0], 'radius': 6},  # 鼻子
    11: {'name': '7', 'color': [0, 255, 0], 'radius': 6},  # 尾巴
    12: {'name': '8', 'color': [0, 255, 0], 'radius': 6},  # 脖子
}
    kpt_labelstr = {
    'font_size': 1.5,  # 字体大小
    'font_thickness': 3,  # 字体粗细
    'offset_x': 10,  # X 方向，文字偏移距离，向右为正
    'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
}
    skeleton_map = [

    {'srt_kpt_id': 0, 'dst_kpt_id': 4, 'color': [0, 100, 255], 'thickness': 1},
    {'srt_kpt_id': 1, 'dst_kpt_id': 5, 'color': [0, 255, 0], 'thickness': 1},
    {'srt_kpt_id': 2, 'dst_kpt_id': 6, 'color': [255, 0, 0], 'thickness': 1},
    {'srt_kpt_id': 3, 'dst_kpt_id': 7, 'color': [0, 0, 255], 'thickness': 1},

    {'srt_kpt_id': 4, 'dst_kpt_id': 12, 'color': [148, 0, 69], 'thickness': 1},
    {'srt_kpt_id': 5, 'dst_kpt_id': 12, 'color': [47, 255, 173], 'thickness': 1},
    {'srt_kpt_id': 6, 'dst_kpt_id': 11, 'color': [203, 192, 255], 'thickness': 1},
    {'srt_kpt_id': 7, 'dst_kpt_id': 11, 'color': [86, 0, 25], 'thickness': 1},

    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [122, 160, 255], 'thickness': 1},
    {'srt_kpt_id': 9, 'dst_kpt_id': 10, 'color': [139, 0, 139], 'thickness': 1},

    {'srt_kpt_id': 10, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 1},

    {'srt_kpt_id': 12, 'dst_kpt_id': 11, 'color': [152, 251, 152], 'thickness': 1},

    ]
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    for idx in range(num_bbox):  # 遍历每个框
        bbox_xyxy = bboxes_xyxy[idx]
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)
        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])
        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = round(bbox_keypoints[srt_kpt_id][0])
            srt_kpt_y = round(bbox_keypoints[srt_kpt_id][1])
            srt_kpt_conf = bbox_keypoints[srt_kpt_id][2]  # 获取起始点置信度
            # print(srt_kpt_conf)
            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = round(bbox_keypoints[dst_kpt_id][0])
            dst_kpt_y = round(bbox_keypoints[dst_kpt_id][1])
            dst_kpt_conf = bbox_keypoints[dst_kpt_id][2]  # 获取终止点置信度
            # print(dst_kpt_conf)
            # 获取骨架连接颜色
            skeleton_color = skeleton['color']
            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']
            # 如果起始点和终止点的置信度都高于阈值，才画骨架连接
            if srt_kpt_conf > 0.5 and dst_kpt_conf > 0.5:
                # 画骨架连接
                img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                                   thickness=skeleton_thickness)
        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = round(bbox_keypoints[kpt_id][0])
            kpt_y = round(bbox_keypoints[kpt_id][1])
            kpt_conf = bbox_keypoints[kpt_id][2]  # 获取该关键点置信度
            if kpt_conf > 0.5:
                # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
           # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1 / (end_time - start_time)

        # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
        FPS_string = 'FPS  {:.2f}'.format(FPS)  # 写在画面上的字符串
        img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
    return img_bgr


def generate_video(input_path=None, weight=None):
    filehead = input_path.split('/')[-1]
    output_path = "{}-out-".format(weight.split('/')[-1].split(".")[-2]) + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                # 处理帧
                # frame_path = './temp_frame.png'
                # cv2.imwrite(frame_path, frame)
                try:
                    frame = run_to_yolo(frame,weight)
                except:
                    print('error')
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)


generate_video(input_path='./test/一只老虎.mp4', weight=weight)

# cv2.imwrite("{}-out-".format(weight.split('/')[-1].split(".")[-2]) + "test/000000.jpg".split('/')[-1],run_to_yolo(cv2.imread("test/000000.jpg"),weight=weight))

