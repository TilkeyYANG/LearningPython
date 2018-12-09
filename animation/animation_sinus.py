# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:51:08 2018

@author: TilkeyYANG
"""



# Caution: in prompt do: conda install -c conda-forge ffmpeg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import os

# CD
os.chdir('.')
cwd = os.getcwd()
print('Working Directory:', cwd)
os.makedirs(cwd + '/anime_output', exist_ok=True)

# Create subplot
fig, ax = plt.subplots()

# Create x and y data
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

# define animate fonction, to updating x and y value. i means the current feame：
def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,
  
def init():
    line.set_ydata(np.sin(x))
    return line,
  
# Set up format for saving the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)
# =============================================================================
#     fig - figureto to be animated
#     func - defined automation function
#     frames - frame number
#     init_func - starting frame
#     interval - update frequnce
#     blit - refresh all points or only changed points
# =============================================================================

plt.show()
os.chdir('./anime_output')
ani.save('animation_sin.mp4', writer=writer)