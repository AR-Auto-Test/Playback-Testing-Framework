"""import os

# 输入文件路径
input_file = "data_video_list.txt"
# 输出文件路径
output_file = "gt.txt"
# 视频文件夹路径
video_folder = "video_clips"

# 读取输入文件
with open(input_file, "r") as file:
    lines = file.readlines()

# 创建或清空输出文件
open(output_file, "w").close()

# 逐行处理
for line in lines:
    line = line.strip()
    if ":" in line:
        filename, label = line.split(":")
        if label == "keep":
            # 如果标签为keep，将文件名写入输出文件
            with open(output_file, "a") as file:
                file.write(filename + ":\n")
        elif label == "delete":
            # 如果标签为delete，删除对应的视频文件
            video_path = os.path.join(video_folder, filename)
            if os.path.exists(video_path):
                os.remove(video_path)
            else:
                print(f"文件不存在: {video_path}")
        else:
            print(f"无效的标签: {label}")
    else:
        print(f"无效的行格式: {line}")

print("处理完成")"""

import os

# 定义文件夹路径和输出文件名
folder_path = "video_clips_6s_2"
output_file = "pre_process/gt_llm.txt"

# 打开输出文件
with open(output_file, "w", encoding="utf-8") as f_out:
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为mp4格式
        if file_name.endswith(".mp4"):
            # 去掉后缀并添加冒号
            clean_name = os.path.splitext(file_name)[0] + ":"
            # 写入输出文件
            f_out.write(clean_name + "\n")

print(f"文件名已写入 {output_file} 中。")
