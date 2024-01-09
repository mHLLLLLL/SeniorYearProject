import torch
from ultralytics import YOLO
import onnx
import time
import numpy as np
import cv2
import onnxruntime
from ultralytics.utils import ops
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model_path = "weight/tiger-x-sim.onnx"

# 绘制相关参数
kpts_shape = [13, 3]  # 关键点 shape
bbox_name = 'tiger'

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

ort_session = onnxruntime.InferenceSession(model_path,
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                           )
model_input = ort_session.get_inputs()
input_shape = model_input[0].shape
input_name = model_input[0].name
input_height, input_width = input_shape[2:]
output_name = ort_session.get_outputs()[0].name


def run_to_onnx(img_bgr, x_ratio, y_ratio):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    # 记录该帧开始处理的时间
    start_time = time.time()

    # 预处理-缩放图像尺寸
    img_bgr_640 = cv2.resize(img_bgr, [input_height, input_width])
    img_rgb_640 = img_bgr_640[:, :, ::-1]
    # 预处理-归一化
    input_tensor = img_rgb_640 / 255
    # 预处理-构造输入 Tensor
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 加 batch 维度
    input_tensor = input_tensor.transpose((0, 3, 1, 2))  # N, C, H, W
    input_tensor = np.ascontiguousarray(input_tensor)  # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float()  # 转 Pytorch Tensor
    # input_tensor = input_tensor.half() # 是否开启半精度，即 uint8 转 fp16，默认转 fp32 

    # ONNX Runtime 推理预测
    ort_output = ort_session.run([output_name], {input_name: input_tensor.cpu().numpy()})[0]
    # 转 Tensor
    preds = torch.Tensor(ort_output)

    # 后处理-置信度过滤、NMS过滤
    preds = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, nc=1)
    pred = preds[0]

    # 解析目标检测预测结果
    pred_det = pred[:, :6].cpu().numpy()
    num_bbox = len(pred_det)
    bboxes_cls = pred_det[:, 5]  # 类别
    bboxes_conf = pred_det[:, 4]  # 置信度

    # 目标检测框 XYXY 坐标
    # 还原为缩放之前原图上的坐标
    pred_det[:, 0] = pred_det[:, 0] * x_ratio
    pred_det[:, 1] = pred_det[:, 1] * y_ratio
    pred_det[:, 2] = pred_det[:, 2] * x_ratio
    pred_det[:, 3] = pred_det[:, 3] * y_ratio
    pred_det[pred_det < 0] = 0  # 把小于零的抹成零
    bboxes_xyxy = pred_det[:, :4].astype('uint32')

    # 解析关键点检测预测结果
    pred_kpts = pred[:, 6:].view(len(pred), kpts_shape[0], kpts_shape[1])
    bboxes_keypoints = pred_kpts.cpu().numpy()
    # 还原为缩放之前原图上的坐标
    bboxes_keypoints[:, :, 0] = bboxes_keypoints[:, :, 0] * x_ratio
    bboxes_keypoints[:, :, 1] = bboxes_keypoints[:, :, 1] * y_ratio
    bboxes_keypoints = bboxes_keypoints.astype('uint32')

    # OpenCV可视化
    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # pdb.set_trace()

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = bbox_name

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
            srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
            srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
            dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

            # 获取骨架连接颜色
            skeleton_color = skeleton['color']

            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']

            # 画骨架连接
            img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                               thickness=skeleton_thickness)

        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            # kpt_label = str(kpt_id) # 写关键点类别 ID（二选一）
            kpt_label = str(kpt_color_map[kpt_id]['name'])  # 写关键点类别名称（二选一）
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  {:.2f}'.format(FPS)  # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    return img_bgr


def run_to_onnx_cam():
    # 获取摄像头，传入0表示获取系统默认摄像头
    cap = cv2.VideoCapture(1)

    # 打开cap
    cap.open(0)

    # 无限循环，直到break被触发
    while cap.isOpened():

        # 获取画面
        success, frame = cap.read()
        # X 方向 图像缩放比例
        x_ratio = frame.shape[1] / input_width
        # Y 方向 图像缩放比例
        y_ratio = frame.shape[0] / input_height

        if not success:  # 如果获取画面不成功，则退出
            print('获取画面不成功，退出')
            break

        ## 逐帧处理
        try:
            frame = run_to_onnx(frame, x_ratio, y_ratio)
        except:
            pass

        # 展示处理后的三通道图像
        cv2.imshow('my_window', frame)

        key_pressed = cv2.waitKey(60)  # 每隔多少毫秒毫秒，获取键盘哪个键被按下
        # print('键盘上被按下的键：', key_pressed)

        if key_pressed in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    # 关闭摄像头
    cap.release()

    # 关闭图像窗口
    cv2.destroyAllWindows()


def run_to_onnx_video(input_path=None, weight=None):
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
                x_ratio = frame.shape[1] / input_width
                # Y 方向 图像缩放比例
                y_ratio = frame.shape[0] / input_height
                if not success:
                    break

                # 处理帧
                # frame_path = './temp_frame.png'
                # cv2.imwrite(frame_path, frame)
                try:
                    frame = run_to_onnx(frame, x_ratio, y_ratio)
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


def run_to_onnx_img(input_path=None, weight=None):
    filehead = input_path.split('/')[-1]
    output_path = "{}-out-".format(weight.split('/')[-1].split(".")[-2]) + filehead

    result = cv2.imread(input_path)
    x_ratio = result.shape[1] / input_width
    # Y 方向 图像缩放比例
    y_ratio = result.shape[0] / input_height
    img = run_to_onnx(result, x_ratio, y_ratio)

    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    # run_to_onnx_cam()
    run_to_onnx_video(input_path='./test/一只老虎.mp4',weight=model_path)
    # run_to_onnx_img(input_path='./test/000000.jpg',weight=model_path)
