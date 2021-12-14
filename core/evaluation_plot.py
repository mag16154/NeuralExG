
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import animation
import sys
from itertools import product, combinations
import os


def plotInvSenResults(trajectories, destinations, d_time_step, dimensions, best_trajectory, x_vals=None, xp_vals=None):
    n_trajectories = len(trajectories)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    destination = destinations[0]

    if dimensions == 3:
        x_index = 0
        y_index = 1
        z_index = 2
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')
        ax.plot3D(trajectories[0][:, x_index], trajectories[0][:, y_index], trajectories[0][:, z_index],
                  color=colors[7], label='Reference trajectory')
        ax.scatter3D(trajectories[0][0, x_index], trajectories[0][0, y_index], trajectories[0][0, z_index],
                     color='green', label='states at time 0')
        ax.scatter3D(trajectories[0][d_time_step, x_index], trajectories[0][d_time_step, y_index],
                     trajectories[0][d_time_step, z_index], color='red', label='states at time t')

        if x_vals is not None:
            for idx in range(len(x_vals)):
                x_val = x_vals[idx]
                xp_val = xp_vals[idx]
                ax.scatter3D(x_val[x_index], x_val[y_index], x_val[z_index], color='green')
                ax.scatter3D(xp_val[x_index], xp_val[y_index], xp_val[z_index], color='red')
        else:
            for idx in range(1, n_trajectories - 1):
                trajectory = trajectories[idx]
                pred_init = trajectory[0]
                pred_destination = trajectory[d_time_step]
                ax.scatter3D(pred_init[x_index], pred_init[y_index], pred_init[z_index], color='green')
                ax.scatter3D(pred_destination[x_index], pred_destination[y_index], pred_destination[z_index],
                             color='red')

        ax.plot3D(best_trajectory[:, x_index], best_trajectory[:, y_index],
                  best_trajectory[:, z_index], color='blue', label='Final trajectory')

        ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black',
                     label='Destination z')

        # For plotting cube initial set
        # r = [0.2, 0.5]
        # for s, e in combinations(np.array(list(product(r, r, r))), 2):
        #     if np.sum(np.abs(s - e)) == r[1] - r[0]:
        #         ax.plot3D(*zip(s, e), color="black")
        # ax.set_title("Cube")

        for destination in destinations:
            ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black')
        plt.legend()
        plt.show()

    else:
        plt.figure(1)
        x_indices = [0]
        y_indices = [1]

        # To plot an obstacle
        # c_center_x = 1.75
        # c_center_y = -0.25
        # c_size = 0.1
        # deg = list(range(0, 360, 5))
        # deg.append(0)
        # xl = [c_center_x + c_size * math.cos(np.deg2rad(d)) for d in deg]
        # yl = [c_center_y + c_size * math.sin(np.deg2rad(d)) for d in deg]
        # plt.plot(xl, yl, color='k')

        if dimensions == 4 or dimensions == 5:
            x_indices = [0, 1]
            y_indices = [3, 2]
        elif dimensions == 6:
            # ACC 3L: 0, 5, 1, 4, 2, 3
            # ACC 5L: 0, 5, 3, 2, 1, 4
            # ACC 7L, 10L: 0, 5, 1, 4, 2, 3
            x_indices = [0, 3, 1]
            y_indices = [5, 2, 4]
        elif dimensions == 12:
            x_indices = [0, 2, 4, 6, 8, 10]
            y_indices = [1, 3, 5, 7, 9, 11]

        for x_idx in range(len(x_indices)):
            x_index = x_indices[x_idx]
            y_index = y_indices[x_idx]
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7],
                     label='Reference Trajectory')
            # starting_state = trajectories[0][d_time_step]
            plt.plot(trajectories[0][0, x_index], trajectories[0][0, y_index], 'g^', label='states at time 0')
            plt.plot(trajectories[0][d_time_step, x_index], trajectories[0][d_time_step, y_index], 'r^',
                     label='states at time t')

            if x_vals is not None:
                for idx in range(len(x_vals)):
                    x_val = x_vals[idx]
                    xp_val = xp_vals[idx]
                    plt.plot(x_val[x_index], x_val[y_index], 'g^')
                    plt.plot(xp_val[x_index], xp_val[y_index], 'r^')
                for idx in range(1, n_trajectories - 1):
                    trajectory = trajectories[idx]
                    plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

            else:
                for idx in range(1, n_trajectories - 1):
                    trajectory = trajectories[idx]
                    pred_init = trajectory[0]
                    pred_destination = trajectory[d_time_step]
                    plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                    plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')
                    # plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

            plt.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], 'g', label='Intermediate trajectory')
            plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b', label='Final trajectory')

            plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

            for destination in destinations:
                plt.plot(destination[x_index], destination[y_index], 'ko')
            plt.legend()
            plt.show()


def plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory, usafeupperBoundArray, usafelowerBoundArray,
                             data_object):

    x_index = 0
    y_index = 1
    u_x_min = usafelowerBoundArray[x_index]
    u_x_max = usafeupperBoundArray[x_index]
    u_y_min = usafelowerBoundArray[y_index]
    u_y_max = usafeupperBoundArray[y_index]

    u_verts = [
            (u_x_min, u_y_min),  # left, bottom
            (u_x_max, u_y_min),  # left, top
            (u_x_max, u_y_max),  # right, top
            (u_x_min, u_y_max),  # right, bottom
            (u_x_min, u_y_min),  # ignored
    ]

    i_x_min = data_object.lowerBoundArray[x_index]
    i_x_max = data_object.upperBoundArray[x_index]
    i_y_min = data_object.lowerBoundArray[y_index]
    i_y_max = data_object.upperBoundArray[y_index]

    i_verts = [
            (i_x_min, i_y_min),  # left, bottom
            (i_x_max, i_y_min),  # left, top
            (i_x_max, i_y_max),  # right, top
            (i_x_min, i_y_max),  # right, bottom
            (i_x_min, i_y_min),  # ignored
        ]

    codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
    ]

    u_path = Path(u_verts, codes)

    fig, ax = plt.subplots()

    u_patch = patches.PathPatch(u_path, facecolor='red', lw=1)
    ax.add_patch(u_patch)

    i_path = Path(i_verts, codes)

    i_patch = patches.PathPatch(i_path, facecolor='red', lw=1, fill=False)
    ax.add_patch(i_patch)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    # colors = ['red', 'black', 'blue', 'brown', 'green']

    n_trajectories = len(trajectories)
    ax.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7], label='Reference trajectory')

    for idx in range(1, n_trajectories - 1):
        pred_init = trajectories[idx][0]
        ax.scatter(pred_init[x_index], pred_init[y_index], color='g')
        pred_dest = trajectories[idx][d_time_step]
        ax.scatter(pred_dest[x_index], pred_dest[y_index], color='r')
        ax.plot(trajectories[idx][:, x_index], trajectories[idx][:, y_index], color='g')

    ax.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], color='g', label='Course correction')
    ax.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], color=colors[1], label='Falsifying trajectory')

    ax.set_xlabel('x' + str(x_index))
    ax.set_ylabel('x' + str(y_index))
    # ax.set_xlim(5, 10)
    # ax.set_ylim(5, 30)
    plt.legend()

    plt.show()


# https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
# https://matplotlib.org/stable/api/animation_api.html
# https://matplotlib.org/stable/gallery/animation/dynamic_image.html
# http://blog.vikramank.com/2015/02/methods-animations-loop-animation-package-python/
# https://stackoverflow.com/questions/23049762/matplotlib-multiple-animate-multiple-lines
# https://stackoverflow.com/questions/38980794/python-matplotlib-funcanimation-save-only-saves-100-frames
# https://stackoverflow.com/questions/58263646/attributeerror-list-object-has-no-attribute-get-zorder
# http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
def plotInvSenResultsAnimate(trajectories, destinations, d_time_step, v_vals= None, vp_vals=None):
    n_trajectories = len(trajectories)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    fig = plt.figure()
    # ax1 = plt.axes(xlim=(-0.5, 1.5), ylim=(-1.5, 1.5))  # For bench1
    ax1 = plt.axes(xlim=(0.4, 1.8), ylim=(-0.7, 1.25))  # For bench2
    # ax1 = plt.axes(xlim=(-1.0, 0.75), ylim=(-0.45, -0.15))  # For bench9-III (Tanh)
    # ax1 = plt.axes(xlim=(-0.5, 0.5), ylim=(-0.5, 1.0))  # For bench9

    plt.xlabel('X')
    plt.ylabel('Y')

    lines = []
    dots = []
    vectors = []
    x_xp_dots = []
    destination = destinations[0]

    for index in range(n_trajectories):
        lobj = ax1.plot([], [], lw=1, color=colors[7])[0]
        lines.append(lobj)

    vobj = ax1.plot([], [], lw=3, color='red')[0]
    vectors.append(vobj)
    vobj = ax1.plot([], [], lw=3, color='red')[0]
    vectors.append(vobj)
    # ax1.cla()
    # vobj = ax1.arrow(0.0, 0.0, 0.0, 0.0,  head_width=2, head_length=2, fc='black', ec='black')
    # vectors.append(vobj)
    # vobj = ax1.arrow(0.0, 0.0, 0.0, 0.0,  head_width=2, head_length=2, fc='black', ec='black')
    # vectors.append(vobj)
    dotObj = ax1.plot([], [], 'ko', label='destination')[0]  # destination
    dots.append(dotObj)
    dotObj = ax1.plot([], [], color='black', marker="*", markersize=6)[0]  # next state at 0
    dots.append(dotObj)
    dotObj = ax1.plot([], [], color='black', marker="*", markersize=6)[0]  # next state at d_time_step
    dots.append(dotObj)

    dObj = ax1.plot([], [], 'b.', label='initial state')[0]  # state at 0
    x_xp_dots.append(dObj)
    dObj = ax1.plot([], [], 'g.', label='state at time t')[0]  # state at d_time_step
    x_xp_dots.append(dObj)

    for index in range(n_trajectories-1):
        dObj = ax1.plot([], [], 'b.')[0]  # state at 0
        x_xp_dots.append(dObj)
        dObj = ax1.plot([], [], 'g.')[0]  # state at d_time_step
        x_xp_dots.append(dObj)

    def init():
        for line in lines:
            line.set_data([], [])
        for dot in dots:
            dot.set_data([], [])
        for vec in vectors:
            vec.set_data([], [])
        for dot in x_xp_dots:
            dot.set_data([], [])
        return lines, dots, vectors, x_xp_dots

    frame_num = n_trajectories

    # test 1 - Bench2
    x_index = 0
    y_index = 1

    # test 2 - Bench9 Tanh
    # x_index = 0
    # y_index = 3

    # test 3 - Bench 9
    # x_index = 2
    # y_index = 3

    def animate(idx):
        graph_list = []
        # sc_1.set_offsets([destination[x_index], destination[y_index]])
        trajectory = trajectories[idx]

        # dots[1].set_data(trajectory[0, x_index], trajectory[0, y_index])
        # dots[2].set_data(trajectory[d_time_step, x_index], trajectory[d_time_step, y_index])
        dots[0].set_data(destination[x_index], destination[y_index])
        xlist = [trajectory[:, x_index]]
        ylist = [trajectory[:, y_index]]
        lines[idx].set_data(xlist, ylist)
        x_xp_dots[2*idx].set_data(trajectory[0, x_index], trajectory[0, y_index])
        x_xp_dots[2*idx+1].set_data(trajectory[d_time_step, x_index], trajectory[d_time_step, y_index])
        # for lnum, line in enumerate(lines):
        #     line.set_data(xlist, ylist)  # set data for each line separately.

        # bench 9 and 9-tanh
        scale_v = 5.0
        scale_vp = 6.0
        if idx < frame_num-1:
            xlist = [trajectory[0, x_index]]
            xlist.append(trajectory[0, x_index] + scale_v * v_vals[idx+1][x_index])
            # xlist.append(0.1 * x_vals[idx+1][x_index])
            ylist = [trajectory[0, y_index]]
            ylist.append(trajectory[0, y_index] + scale_v * v_vals[idx+1][y_index])
            # xlist.append(0.1 * x_vals[idx+1][y_index])
            vectors[0].set_data(xlist, ylist)

            xlist = [trajectory[d_time_step, x_index]]
            xlist.append(trajectory[d_time_step, x_index] + scale_vp * vp_vals[idx][x_index])
            # xlist.append(xp_vals[idx + 1][x_index])
            ylist = [trajectory[d_time_step, y_index]]
            ylist.append(trajectory[d_time_step, y_index] + scale_vp * vp_vals[idx][y_index])
            # ylist.append(xp_vals[idx + 1][y_index])
            vectors[1].set_data(xlist, ylist)

            # dots[1].set_data(x_vals[idx+1][x_index], x_vals[idx+1][y_index])
            dots[1].set_data(trajectory[0, x_index] + scale_v * v_vals[idx+1][x_index] , trajectory[0, y_index] + scale_v * v_vals[idx+1][y_index])
            dots[2].set_data(trajectory[d_time_step, x_index] + scale_vp * vp_vals[idx][x_index], trajectory[d_time_step, y_index] + scale_vp * vp_vals[idx][y_index])
            # dots[2].set_data(xp_vals[idx + 1][x_index], xp_vals[idx + 1][y_index])
        # else:
        #     xlist = []
        #     ylist = []
        #     vectors[0].set_data(xlist, ylist)
        #     vectors[1].set_data(xlist, ylist)
        #     dots[1].set_data([], [])
        #     dots[2].set_data([], [])

        graph_list.append(lines)
        graph_list.append(dots)
        graph_list.append(x_xp_dots)
        graph_list.append(vectors)
        graph_list = [item for sublist in graph_list for item in sublist]
        legend = plt.legend()
        return graph_list, legend

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_num, interval=500, save_count=sys.maxsize)

    # def update(idx, trajectories, line):
    #     line.set_data(trajectories[:, x_index], y1[:num])
    #     return line,
    #
    # ani = animation.FuncAnimation(fig, update, frame_num, fargs=[trajectories, line, ], interval=200, blit=True)

    # anim.save('test.mp4', fps=5.0, dpi=200)
    anim.save('test.gif', writer='imagemagick', fps=4)
    # plt.legend()
    # plt.show()

