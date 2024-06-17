import matplotlib.pyplot as plt
import numpy as np
 
 
import matplotlib.pyplot as plt

x = np.array([1,3,4,5])
y = np.array([10,3,5,4])
z = np.array([10,3,11,-3])

figure, axis = plt.subplots(1,2) 

axis[0].plot(x,y)
axis[1].plot(x,z)

plt.show()