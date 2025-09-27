# ecg-signal-analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

本仓库主要记录ECG相关实践过程与总结。

## 1. 标准流程


![ECG分析流程图](/preprocessing.PNG) 
1.  **数据加载**: 使用 `wfdb` 库读取 MIT-BIH 数据库的原始信号（`.dat`）和专家标注（`.atr`）。
2.  **信号预处理**:
    * **带通滤波**: 滤除基线漂移和高频噪声。
    * **小波去噪**: 进一步提纯信号，保留心搏的关键波形特征。
3.  **特征工程**:
    * **R波检测**: 采用峰值检测算法，精确定位每次心搏的核心—R波。
    * **心搏切片**: 以R波为中心，截取固定长度的信号片段，作为心搏的波形形态特征。
    * **HRV特征提取**: 计算心率变异性（HRV）指标，如SDNN、RMSSD等，作为心搏的节律特征。
4.  **模型构建与评估**:
    * **传统机器学习**: 基于HRV特征，训练并评估SVM、随机森林等模型。
    * **深度学习**: 基于心搏波形切片，训练并对比CNN和RNN模型的性能。

---

## 2. 可复用代码模块

### 模块一：`wfdb` 数据加载范式

这是读取MIT-BIH数据库中任意记录（record）的标准代码。

```python
import wfdb

def load_ecg_record(record_name, data_dir):
    """
    加载指定ECG记录的信号和注释。
    """
    record_path = f"{data_dir}/{record_name}"
    # 读取信号，通常选择MLII导联
    record = wfdb.rdrecord(record_path, channels=[0])
    signals = record.p_signal.flatten()
    fs = record.fs
    
    # 读取注释
    annotation = wfdb.rdann(record_path, 'atr')
    annotation_times = annotation.sample
    annotation_symbols = annotation.symbol
    
    return signals, annotation, fs

# 示例
# signals, annotation, fs = load_ecg_record("100", "/path/to/mit-bih/")
```

### 模块二：信号预处理范式

结合带通滤波和小波变换，可以得到高质量的ECG信号用于后续分析。

```python
from scipy import signal
import pywt

# 带通滤波
def bandpass_filter(signals, fs, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signals)

# 小波去噪
def wavelet_denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    # ... (阈值处理) ...
    return pywt.waverec(coeffs=coeffs, wavelet='db5')
```

**效果**:
![图片](/1.png) 

 ### 模块三：R波检测与HRV特征提取范式

该模块用于从预处理后的信号中定位心搏并提取其节律特征。

```python
# R波检测
def detect_r_peaks(signals, fs):
    min_distance = int(0.2 * fs) # 假设心率不高于300bpm
    peaks, _ = signal.find_peaks(signals, distance=min_distance, height=np.mean(signals))
    return peaks

# HRV特征提取
def extract_hrv_features(peaks, fs):
    rr_intervals = np.diff(peaks) / fs  # 计算RR间期（秒）
    
    sdnn = np.std(rr_intervals)  # RR间期标准差
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) # 相邻RR间期差的均方根
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) # 差异大于50ms的次数
    pnn50 = nn50 / len(np.diff(rr_intervals)) if len(np.diff(rr_intervals)) > 0 else 0
    
    return {'sdnn': sdnn, 'rmssd': rmssd, 'pnn50': pnn50}

# peaks = detect_r_peaks(filtered_signals, fs)
# hrv_features = extract_hrv_features(peaks, fs)
```

**R波检测效果**:
![图片](/2.png) 
**RR间期检测**:
![图片](/3.png) 


 ---

## 3\. 实验与发现

### 实验一：基于HRV特征的机器学习分类

  - **方法**: 对每个心搏提取HRV特征，并使用**随机森林**、**SVM**和**LightGBM**进行分类。
  - **发现**: 在仅使用HRV时域特征的情况下，**随机森林**表现最佳。然而，这种方法丢失了心搏的形态信息，对某些特定心律失常类型的区分能力有限。
  ![图片](/4.png) 
### 实验二：基于波形特征的深度学习分类

  - **方法**: 直接使用R波周围的信号切片作为输入，对比**1D-CNN** (`CNN_pm.ipynb`) 和**RNN** (`RNN_pm.ipynb`) 的性能。
  - **发现**:
      - **CNN** 在此任务中表现极为出色（测试集准确率 **99.46%**），证明了其在从一维序列中捕捉“形态模式”的强大能力。
      - **RNN** 也取得了很好的性能（测试集准确率 **96.52%**），验证了其处理序列数据的有效性。
      - **对比分析**: 对于单次心搏的分类，CNN更占优势。但对于需要长时依赖的复杂心律失常（如房颤），RNN的理论优势可能更大。

## 4\. 代码文件列表

  - `ECG_MS1.ipynb`: 数据分析、预处理、HRV特征提取与机器学习模型初步实践。
  - `CNN_pm.ipynb`: 使用TensorFlow实现的1D-CNN心搏分类模型。
  - `RNN_pm.ipynb`: 使用PyTorch实现的RNN心搏分类模型。

-----

## 许可证 (License)

本仓库采用 [MIT 许可证](LICENSE)。

