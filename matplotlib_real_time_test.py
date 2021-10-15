import matplotlib.pyplot as plt
import numpy as np
import math
plt.ion() ## Note this correction
fig=plt.figure()
plt.axis([0,10,-1.5,1.5])
plt.axis()
i=0
x=list()
y=list()

while i <10:
    temp_y=math.sin(i)
    x.append(i)
    y.append(temp_y)
    plt.scatter(i,temp_y, s=0.5, alpha=1)
    i+=0.01
    plt.show()
    plt.pause(0.00001) #Note this correction