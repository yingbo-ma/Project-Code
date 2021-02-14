import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y2 = [.50, .58, .62, .58, .40, .60, .54, .62, .62, .56]
y1 = [.49, .52, .54, .51, .43, .56, .50, .58, .57, .47]

a, = plt.plot(x, y1, 'r--', label="acoustic feature")
b, = plt.plot(x, y2, 'b--', label="combined feature")

first_legend = plt.legend(handles=[a], loc='upper right')
ax = plt.gca().add_artist(first_legend)


plt.legend(handles=[b], loc='lower right')

plt.axis([1, 10, 0, 1])
plt.show()