# COVID-19：使用深度学习的医学诊断
南華大學_跨領域-人工智慧_MidReport

11124208王品雯、11124209蔡岱伶
介绍
正在进行的名为 COVID-19 的全球大流行是由 SARS-COV-2 引起的，该病毒传播迅速并发生变异，引发了几波疫情，主要影响第三世界和发展中国家。随着世界各国政府试图控制传播，受影响的人数正在稳步上升。
![期末截圖01]()
本文将使用 CoronaHack-Chest X 射线数据集。它包含胸部 X 射线图像，我们必须找到受冠状病毒影响的图像。
我们之前谈到的 SARS-COV-2 是主要影响呼吸系统的病毒类型，因此胸部 X 射线是我们可以用来识别受影响肺部的重要成像方法之一。这是一个并排比较：
![期末截圖02]()
如你所见，COVID-19 肺炎如何吞噬整个肺部，并且比细菌和病毒类型的肺炎更危险。
本文，将使用深度学习和迁移学习对受 Covid-19 影响的肺部的 X 射线图像进行分类和识别。
## 导入库和加载数据
    import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import pandas as pd
sns.set()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50


import PIL.Image
import matplotlib.pyplot as mpimg
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image


from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


from sklearn.utils import shuffle


train_df = pd.read_csv('../input/corona hack-chest-xray dataset/Chest_xray_Corona_Metadata.csv')
train_df.shape
> (5910, 6)


train_df.head(5)



