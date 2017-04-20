import numpy as np
from scipy.signal import convolve2d as conv

a = [[3,1,4,1,5],[9,2,6,5,3],[5,8,9,7,9],[3,2,3,8,4],[6,2,6,4,3]]
a = np.array(a)
b = [[3,8,3],[2,7,9],[5,0,2]]
b = np.array(b)
c = conv(a,b)
print(c)