import torch
import numpy as np
from PIL import Image
#import accimage

# img = Image.open('./2007_000032.png')
# img = np.array(img)
# print(img.shape)

# pic = np.ones([34,34])
# img = torch.from_numpy(pic.transpose((2, 0, 1)))
# print(img.size())

# mask = np.ones([3,5])
# boolmask = (mask>=0) and (mask<2) #这种是错误的
# print(boolmask)

a1 = np.zeros([3,15,15])
b1 = np.ones([3,15,15])
a1[2, 5:9, 5:9] = b1[1, 5:9, 5:9]
a1 /= b1

#print(a1)
tensor1 = torch.from_numpy(a1)
print(tensor1)
print(tensor1.max(0)[1].squeeze_(0).numpy())