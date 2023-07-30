if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero 
from dezero import Variable
import dezero.functions as F
from dezero.models import VGG16
from PIL import Image


# 1
# model = VGG16(pretrained=True)

# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
# model.plot(x, to_file='./steps/step58_plot.png') # 계산 그래프 시각화

# 2
# url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg'
# img_path = dezero.utils.get_file(url)
# img = Image.open(img_path)
# # img.show()

# x = VGG16.preprocess(img)
# print(type(x), x.shape)

# 3
url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis] # 배치용 축 추가

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='./steps/step58_vgg.pdf') # 계산 그래프 시각화
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
