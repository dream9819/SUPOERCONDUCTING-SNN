import numpy as np
import librosa as rosa


class Feature_Extraction:
    def __init__(self, fmin = 75, n_bins = 25, bins_per_octave = 3, time_step = 40 , compress_style = 'linear' ):
        self.n_bins = n_bins
        self.fmin = fmin
        self.bins = bins_per_octave
        self.time_step = time_step
        self.compress_style = compress_style

    # 常数Q变换,librosa.cqt函数中的Q计算公式为：Q = float(filter_scale) / (2 ** (bins_per_octave) - 1)，可在librosa.cqt_frequencies代码中找到
    # hop_length=512可以理解为间隔512个采样点采一个值，
    def cqt(self, y, sample_rate = 48000):
        return rosa.cqt(y, sr=sample_rate, hop_length=128, n_bins=self.n_bins, fmin=self.fmin,
                       bins_per_octave=self.bins)

    # 各频带中心频率
    def cqt_F(self):
        return rosa.cqt_frequencies(n_bins=self.n_bins, fmin=self.fmin, bins_per_octave=self.bins, tuning=0.0)

    # 各随频率变换的窗口长度
    def cqt_F_width(self, sample_rate):
        return rosa.filters.constant_q_lengths(sample_rate, fmin=self.fmin, n_bins=self.n_bins,
                                              bins_per_octave=self.bins, window='hann', filter_scale=1, gamma=0)

    def sigma_delta(self, input_data , N = 10 , Vt = 0,   pooling = True):
        d = np.zeros((np.array(input_data).shape[0], 1))
        feedback = -1
        integrator = 0
        # 使用池化
        if pooling == True:
            for t in range(len(input_data)):
                i = 0
                if t %self.time_step == 0:
                    feedback = -1
                    integrator = 0
                #for k in range(N*int(input_data[t][1])):
                for k in range(N*2):
                    aq = input_data[t][0] - feedback + integrator  #
                    # 量化编码
                    if aq > Vt:
                        aq_q = 1
                        i += 1
                    else:
                        aq_q = 0
                    if aq_q == 1:
                        # aqq = np.sum(np.maximum(a, 0)) / q  # 语音信号正值平均值
                        # aqq = a[np.argmax(a)]#取输入最大值为量化解码值
                        aqq = 1
                    else:
                        aqq = -1
                    feedback = aqq
                    integrator = aq
                d[t] = i
        # 没有使用池化
        elif pooling == False:
            for t in range(input_data.shape[0]):
                i = 0
                for s in input_data[0][t]:
                    for j in range(N):
                        aq = s - feedback + integrator  #
                        # 量化编码
                        if aq > Vt:
                            aq_q = 1
                            i += 1
                        else:
                            aq_q = 0
                        if aq_q == 1:
                            aqq = 1
                        else:
                            aqq = -1
                        feedback = aqq
                        integrator = aq
                    d[t] = i
            # 输出的d为输入层神经元的输入，每个神经元接收到的为编码为1的数目的总数。
        return d

    def compress(self, data):
        after_compress = [ ]
        if self.compress_style =='linear':
            if data.shape[1]>self.time_step:
                for i in range(data.shape[0]):
                    step = data.shape[1]/self.time_step
                    for num in range(self.time_step):
                        tmp = []
                        avg = np.mean(np.abs(data[i][int(((num)*step//1)):int(((num+1)*step//1))])) # 将np.mean改为np.max(2022年2月5日15:31:47)
                        tmp.append(avg)
                        n = int(((num+1)*step//1)- ((num)*step//1))
                        tmp.append(n)
                        after_compress.append(tmp)
                max = 0 
                min = 1
                for i in after_compress:
                    if i[0] > max:
                            max = i[0]
                    if i[0] <min:
                        min = i[0]
                tmp = max - min
                for i in after_compress:
                    i[0] = 2 * (i[0]-min)/tmp -1
            else :
                for i in range(data.shape[0]):
                    for num in range(data.shape[1]):
                        tmp = []
                        tmp.append(data[num])
                        tmp.append(1)
                        after_compress.append(tmp)
                    for j in range(data.shape[1],self.time_step):
                        tmp = [0,1]
                        after_compress.append(tmp)
                max = 0 
                min = 1
                for i in after_compress:
                    if i[0] > max:
                            max = i[0]
                    if i[0] <min:
                        min = i[0]
                tmp = max - min
                for i in after_compress:
                    i[0] = 2 * (i[0]-min)/tmp -1
        elif self.compress_style == 'sin':
            if data.shape[1]>self.time_step:
                for i in range(data.shape[0]):
                    step = data.shape[1]/self.time_step
                    for num in range(self.time_step):
                        tmps = []
                        tmp = np.mean(np.abs(data[i][int(((num)*step//1)):int(((num+1)*step//1))]))
                        tmps.append(tmp)
                        n = int(((num+1)*step//1)- ((num)*step//1))
                        tmps.append(n)
                        after_compress.append(tmps)
                max = 0 
                min = 1
                for i in after_compress:
                    if i[0] > max:
                            max = i[0]
                    if i[0] <min:
                        min = i[0]
                tmp_max_plus_min = max + min
                tmp_max_minus_min = max - min
                tmp_para = np.pi/(2*tmp_max_minus_min)
                for i in after_compress:
                    i[0] = 2 * i[0] - tmp_max_plus_min
                    i[0] = np.sin(tmp_para*i[0])
            else :
                for i in range(data.shape[0]):
                    for num in range(data.shape[1]):
                        tmp = []
                        tmp.append(data[num])
                        tmp.append(1)
                        after_compress.append(tmp)
                    for j in range(data.shape[1],self.time_step):
                        tmp = [0,1]
                        after_compress.append(tmp)
                max = 0 
                min = 1
                for i in after_compress:
                    if i[0] > max:
                            max = i[0]
                    if i[0] <min:
                        min = i[0]
                tmp_max_plus_min = max + min
                tmp_max_minus_min = max - min
                tmp_para = np.pi/(2*tmp_max_minus_min)
                for i in after_compress:
                    i[0] = 2 * i[0] - tmp_max_plus_min
                    i[0] = np.sin(tmp_para*i[0])
                #after_compress = np.array(after_compress.get())
        return np.array(after_compress)

    def featureextractor(self , files):
        after_feature_extractor = list()
        for file in files:
            after_sigema_delta = []
            y,sr=rosa.load(file,sr=48000)
            after_cqt = self.cqt(y)
            after_compress = self.compress( after_cqt)
            after_sigema_delta = self.sigma_delta(after_compress)
            after_feature_extractor.append( after_sigema_delta )
        return after_feature_extractor
