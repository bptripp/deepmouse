# functional tests of streamlines

import numpy as np
import matplotlib.pyplot as plt
from deepmouse.streamlines import laplace_solution, get_interpolator, get_gradient

# voxel volume with integer spacing, convex and concave parts

size = 20
amplitude = 5

x = np.linspace(0, size, size+1).astype(int)
y = np.linspace(0, size, size+1).astype(int)
z = np.linspace(0, size, size+1).astype(int)
dz = z[1] - z[0]
# x = np.linspace(0, 10, 6)
# y = np.linspace(0, 10, 2)
# z = np.linspace(0, 10, 6)
# bottom = 3+3*np.sin((1/10)*2*np.pi*x)
# top = 17+3*np.sin((1/10)*2*np.pi*x)

# plt.plot(x, bottom)
# plt.plot(x, top)
# plt.show()


X, Y, Z = np.meshgrid(x, y, z)
bottom = (amplitude+amplitude*np.sin((1/size)*2*np.pi*X)).astype(int)
top = (size-amplitude+amplitude*np.sin((1/size)*2*np.pi*X)).astype(int)
in_volume = np.logical_and(Z <= top, Z >= bottom)
on_bottom_edge = np.logical_and(Z >= bottom, Z < bottom+dz)
on_top_edge = np.logical_and(Z <= top, Z > top-dz)

# bottom = Z <= 3+3*np.sin((1/10)*2*np.pi*X),

# print(on_top_edge[0,:,:].any(axis=1))
# print(on_top_edge[0,:,:])
# print(on_bottom_edge[0,:,:].any(axis=1))
# print(on_bottom_edge[0,5,:])

# foo = Z >= bottom
# bar = Z < bottom+dz
# # print(foo[0,5,:])
# # print(bar[0,5,:])
# print(Z[0,5,:])
# print(bottom[0,5,:])

x = X.flatten()
y = Y.flatten()
z = Z.flatten()
in_volume = in_volume.flatten()
on_bottom_edge = on_bottom_edge.flatten()
on_top_edge = on_top_edge.flatten()

ind = np.where(in_volume)
x = x[ind]
y = y[ind]
z = z[ind]
on_bottom_edge = on_bottom_edge[ind]
on_top_edge = on_top_edge[ind]


# print(on_bottom_edge)
# print(ind)
# print(in_volume)

positions = np.concatenate((x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis]), axis=1)

depths = laplace_solution(positions, on_top_edge, on_bottom_edge)


def show_slice(positions, depths):
    slice = []
    for i in range(positions.shape[0]):
        if positions[i,1] == 0:
            slice.append([positions[i,0], positions[i,2], depths[i]])

    slice = np.array(slice)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(slice[:,0], slice[:,1], slice[:,2],
                    cmap='viridis', edgecolor='none')
    plt.plot(slice[:,0], slice[:,1], slice[:,2], 'ko')
    plt.show()


show_slice(positions, depths)

# test interpolation ...
print(type(positions))
print(positions.shape)
# print(depths)
interpolator = get_interpolator(positions, depths)
print(get_gradient(interpolator, [5, 0, 10]))

# fine_size = 40
# fx = np.linspace(0, size, fine_size+1).astype(int)
# fy = np.linspace(0, 1, 2).astype(int)
# fz = np.linspace(0, size, fine_size+1).astype(int)
# FX, FY, FZ = np.meshgrid(fx, fy, fz)
# fx = FX.flatten()
# fy = FY.flatten()
# fz = FZ.flatten()
#
# fdepths = interpolator(fx, fy, fz)
# print('done interpolation')
# fpositions = np.concatenate((fx[:,np.newaxis], fy[:,np.newaxis], fz[:,np.newaxis]), axis=1)
# show_slice(fpositions, fdepths)

