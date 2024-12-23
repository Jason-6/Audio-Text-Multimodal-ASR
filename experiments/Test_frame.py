"""

B2_343 深圳九六三四地面静风洞两左可以落地可以落地深圳九六三四

"""
# import librosa
#
# def get_frames_count(audio_file_path):
# # 读取音频文件
#     y, sr = librosa.load(audio_file_path)
#
#     # 获取语音帧数
#     frames_count = len(y)
#
#     return frames_count
#
#  # 传入音频文件路径
# audio_file_path = '/home/geshuting/Code/CTAL-main/CTAL-main/B2_343.wav'
# frames_count = get_frames_count(audio_file_path)
# #
# print(f"The number of frames in the audio file is: {frames_count}")


#=========================================
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
#
# def visualize_spectrogram(audio_file_path, hop_length=512):
#     # 读取音频文件
#     y, sr = librosa.load(audio_file_path)
#
#     # 计算短时傅里叶变换
#     D = librosa.amplitude_to_db(librosa.stft(y, hop_length=hop_length), ref=np.max)
#
#     # 可视化短时傅里叶变换
#     plt.figure(figsize=(12, 4))
#     librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
#     plt.show()
#
# # 传入音频文件路径
# audio_file_path = '/home/geshuting/Code/CTAL-main/CTAL-main/B2_343.wav'
# visualize_spectrogram(audio_file_path)
#=======================================wave Figure,GET
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
#
# def split_into_frames(audio_file_path, frame_length=0.050, frame_stride=0.0125):
#     # 读取音频文件
#     y, sr = librosa.load(audio_file_path)
#
#     # 计算分帧数目
#     frame_length_samples = int(sr * frame_length)
#     frame_stride_samples = int(sr * frame_stride)
#     frames_count = 1 + int((len(y) - frame_length_samples) / frame_stride_samples)
#
#     # 分帧
#     frames = [y[i*frame_stride_samples : i*frame_stride_samples+frame_length_samples]
#     for i in range(frames_count)]
#
#     return frames, frames_count
#
# # 传入音频文件路径
# audio_file_path = '/home/geshuting/Code/CTAL-main/CTAL-main/B2_343.wav'
# frames, frames_count = split_into_frames(audio_file_path)
#
# # 统计帧序列数量
# print(f"The number of frames in the audio file is: {frames_count}")
#
# # 可视化前几帧的波形图
# y, sr = librosa.load(audio_file_path)
# plt.figure(figsize=(15, 8))
# for i, frame in enumerate(frames[:10]): # 可视化前10帧
#     plt.subplot(10, 1, i+1)
#     librosa.display.waveshow(frame, sr=sr)
#     plt.title(f'Frame {i+1}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#
# plt.tight_layout()
# plt.show()
