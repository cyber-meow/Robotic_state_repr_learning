
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = list(range(10))
y = [y**2 for y in range(10)]
line, = plt.plot(x, y)

def animate(percentage):
    new_y = list(map(lambda a: a*percentage/100, y))
    line.set_data(x, new_y)

ani = animation.FuncAnimation(
    fig, animate, 100, interval=20, blit=False)

ani.save("line_animation.mp4", writer="ffmpeg")
plt.show()
