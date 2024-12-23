import os
import wave

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def get_folder_duration(folder_path):
    total_duration = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                wav_duration = get_wav_duration(file_path)
                total_duration += wav_duration
    return total_duration

folder_path = '/home/geshuting/wav/train'
total_duration = get_folder_duration(folder_path)
print(f'Total duration of {folder_path}: {total_duration/3600:.2f} hours')


####################ok#########################
# import os
# import shutil
#
# # 源文件夹路径
# source_folder = "/home/geshuting/wav/train"
#
# # 目标文件夹路径
# target_folder = "/home/geshuting/wav/train"
#
# # 遍历源文件夹中的所有子文件夹
# for foldername in os.listdir(source_folder):
#     folder_path = os.path.join(source_folder, foldername)
#
#     # 检查是否为文件夹
#     if os.path.isdir(folder_path):
#
#         # 遍历子文件夹中的所有.wav文件
#         for filename in os.listdir(folder_path):
#             if filename.endswith('.wav'):
#                 file_path = os.path.join(folder_path, filename)
#                 target_path = os.path.join(target_folder, filename)
#
#                 # 将.wav文件移动到目标文件夹中
#                 shutil.move(file_path, target_path)


###################spilt transcript############################
# import os
#
# # 设置train文件夹路径、transcript.txt文件路径和txt_files文件夹路径
# train_folder = "/home/geshuting/wav/train"
# transcript_file = "/home/geshuting/aishell_transcript_v0.8.txt"
# txt_folder = "/home/geshuting/wav/train_txt"
#
# # 创建txt_files文件夹（如果它不存在）
# if not os.path.exists(txt_folder):
#     os.mkdir(txt_folder)
#
# # 遍历train文件夹中的所有.wav文件
# for file_name in os.listdir(train_folder):
#     if file_name.endswith(".wav"):
#         # 获取.wav文件的前缀名
#         prefix = os.path.splitext(file_name)[0]
#
#         # 查找与transcript.txt文件中相同前缀名的行
#         with open(transcript_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 fields = line.strip().split()
#                 if fields[0] == prefix:
#                     # 将对应的中文内容写入新的txt文件中
#                     chinese_text = "".join(fields[1:]).replace(" ", "")
#                     new_file_name = os.path.join(txt_folder, prefix + ".txt")
#                     with open(new_file_name, "w", encoding="utf-8") as new_file:
#                         new_file.write(chinese_text)
#                     break
#####################################
# import os
#
# wav_dir = "/home/geshuting/wav/train" # 存放.wav文件的文件夹
# txt_dir = "/home/geshuting/wav/train_txt" # 存放.txt文件的文件夹
#
# # 获取.wav文件和.txt文件的前缀名列表
# wav_prefixes = [os.path.splitext(filename)[0] for filename in os.listdir(wav_dir) if filename.endswith(".wav")]
# txt_prefixes = [os.path.splitext(filename)[0] for filename in os.listdir(txt_dir) if filename.endswith(".txt")]
#
# # 查找在.wav文件夹中存在但在.txt文件夹中不存在的文件
# missing_txt_files = set(wav_prefixes) - set(txt_prefixes)
#
# # 如果存在缺失的文件，则输出到终端
# if missing_txt_files:
#     print("The following files are missing in the .txt folder:")
#     for prefix in missing_txt_files:
#         file_to_delete = os.path.join(wav_dir,prefix+".wav")
#         os.remove(file_to_delete)
#         print("{} delete sucessfully!!!!!!!".format(file_to_delete))
#
# # 查找在.txt文件夹中存在但在.wav文件夹中不存在的文件
# missing_wav_files = set(txt_prefixes) - set(wav_prefixes)
#
# # 如果存在缺失的文件，则输出到终端
# if missing_wav_files:
#     print("The following files are missing in the .wav folder:")
#     for prefix in missing_wav_files:
#         print(prefix + ".txt")
