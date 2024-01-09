# Create your views here.
from django.shortcuts import render, HttpResponse
from app01.models import ModelPath, UploadPath, OutputPath
import os

# inference
from app01.inference_code.inference_oonx import inference
import torch
from ultralytics import YOLO
import time
import numpy as np
import cv2
import onnxruntime
from ultralytics.utils import ops
from tqdm import tqdm


def main(request):
    weight_list = ModelPath.objects.all()
    # 接收文件并判断文件类型
    if request.method == "POST":
        # 接收上传的文件
        upload_file = request.FILES.get("file")

        flag = 0
        # 判断文件类型
        if upload_file.name.endswith(".jpg") or upload_file.name.endswith(".png") or upload_file.name.endswith(".jpeg"):
            use_model_id = request.POST.get("model_name")
            use_model = ModelPath.objects.get(id=use_model_id).path
            print(use_model)



            # 保存文件
            f = open(os.path.join(
                r"D:\OneDrive - 成都东软学院\BaiduSyncdisk\MH-Li_Cloud\University-Folder\NSU\Senior_year_1\AiFw\Project-2\AiWeb\app01\media\upload",
                upload_file.name), mode='wb+')
            for chunk in upload_file.chunks():
                f.write(chunk)
            f.close()
            upload_file_path = os.path.join(
                r"D:\OneDrive - 成都东软学院\BaiduSyncdisk\MH-Li_Cloud\University-Folder\NSU\Senior_year_1\AiFw\Project-2\AiWeb\app01\media\upload",
                upload_file.name)

            run_model = inference(use_model)
            filehead = upload_file_path.split('/')[-1]
            output_path = os.path.join(r"D:\OneDrive - 成都东软学院\BaiduSyncdisk\MH-Li_Cloud\University-Folder\NSU\Senior_year_1\AiFw\Project-2\AiWeb\app01\media\output","{}-out-".format(use_model.split('/')[-1].split(".")[-2]) + filehead)
            run_model.run_to_onnx_img(input_path=upload_file_path, outpath=output_path)

            flag = 1


            # 返回文件路径
            return render(
                request,
                "main.html",
                {"upload_path": f"/media/{upload_file.name},", "weight_list": weight_list, "flag": flag},
            )
        return render(
            request,
            "main.html",
            {"upload_path": f"/media/{upload_file.name},", "weight_list": weight_list},
        )
    return render(
        request,
        "main.html",
        {"weight_list": weight_list}
    )
