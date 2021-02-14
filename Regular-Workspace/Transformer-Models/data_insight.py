
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


x = [0,5,9,10,15]
y = [0,1,2,3,4]

plt.plot(x,y)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
plt.show()