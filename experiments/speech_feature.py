# import numpy as np
# import matplotlib.pyplot as plt
# from  matplotlib.colors import ListedColormap
# # 加载Fbank特征
# fbank_features = np.load("/home/geshuting/dataset/100%_data/20h_parallel/B11_444.npy")
#
# color= ['blue','white']
# cmap = ListedColormap(color)
# # 可视化Mel频谱图
# plt.imshow(fbank_features.T, cmap='jet', aspect='auto')
# plt.xlabel('Frame')
# plt.ylabel('Mel filter')
# plt.title('Mel Spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import librosa

# # 读取wav文件
# waveform, sample_rate = librosa.load("/home/geshuting/dataset/100%_data/20h_parallel/A34_92.wav", sr=None)
#
# # 生成波形图
# plt.figure(figsize=(10, 4))
# plt.plot(np.linspace(0, len(waveform) / sample_rate, num=len(waveform)), waveform)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Waveform')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
fbank_features = np.load('/home/geshuting/dataset/100%_data/20h_parallel/A34_92.npy')

# 绘制谱图
plt.imshow(fbank_features.T, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Frame')
plt.ylabel('FBank Coefficient')
plt.colorbar(label='Magnitude')
plt.title('Spectrogram')
plt.show()
