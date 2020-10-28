from tensorflow.keras.models import load_model
import numpy as np
from frechet import norm
from learningModule import DataConfiguration
from circleRandom import generate_points_in_circle
from os import path
import time
from sampler import generateRandomStates
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from rbflayer import RBFLayer, InitCentersRandom
from matplotlib.path import Path
import matplotlib.patches as patches
import random

# version 1 RRT
from rrtv1 import RRTV1
import math

# version 2 RRT
import sys
sys.path.append('./rrtAlgorithms/src/')
from rrt.rrt import RRT
from search_space.search_space import SearchSpace
from utilities.plotting import Plot
import matplotlib.colors as mcolors


class Evaluation(object):
    def __init__(self, dynamics, dnn_rbf='dnn'):
        self.data_object = None
        self.debug_print = False
        self.dynamics = dynamics
        self.iter_count = 10
        self.n_states = 5
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []
        self.staliro_run = False
        self.eval_dir = '../../eval/'
        self.dnn_rbf = dnn_rbf
        self.rrt_run = False
        self.f_iterations = 0
        self.f_dist = None
        self.f_rel_dist = None

    def getFIterations(self):
        return self.f_iterations

    def getFDistance(self):
        return self.f_dist

    def getFRelDistance(self):
        return self.f_rel_dist

    def setStaliroRun(self):
        self.staliro_run = True

    def getDataObject(self):
        assert self.data_object is not None
        return self.data_object

    def setDataObject(self, d_obj_f_name=None):

        if d_obj_f_name is None:
            d_obj_f_name = self.eval_dir + 'dconfigs/d_object_'+self.dynamics

        d_obj_f_name = d_obj_f_name + '.txt'

        if path.exists(d_obj_f_name):
            d_obj_f = open(d_obj_f_name, 'r')
            lines = d_obj_f.readlines()
            line_idx = 0
            grad_run = bool(lines[line_idx])
            line_idx += 1
            dimensions = int(lines[line_idx])
            line_idx += 1
            steps = int(lines[line_idx][:-1])
            line_idx += 1
            samples = int(lines[line_idx][:-1])
            line_idx += 1
            timeStep = float(lines[line_idx][:-1])
            line_idx += 1
            lowerBoundArray = []
            for idx in range(dimensions):
                token = lines[line_idx][:-1]
                line_idx += 1
                lowerBoundArray.append(float(token))
            upperBoundArray = []
            for idx in range(dimensions):
                token = lines[line_idx][:-1]
                line_idx += 1
                upperBoundArray.append(float(token))
            d_obj_f.close()

            self.data_object = DataConfiguration(dynamics=self.dynamics, dimensions=dimensions)
            self.data_object.setSteps(steps)
            self.data_object.setSamples(samples)
            self.data_object.setTimeStep(timeStep)
            self.data_object.setLowerBound(lowerBoundArray)
            self.data_object.setUpperBound(upperBoundArray)
            if grad_run is True:
                self.data_object.setGradientRun()

        return self.data_object

    def setUnsafeSet(self, lowerBound, upperBound):
        self.usafelowerBoundArray = lowerBound
        self.usafeupperBoundArray = upperBound
        self.staliro_run = True

    def setIterCount(self, iter_count):
        self.iter_count = iter_count

    def setNStates(self, n_states):
        self.n_states = n_states

    def generateRandomUnsafeStates(self, samples):
        states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        return states

    def check_for_bounds(self, state):
        # print("Checking for bounds for the state {}".format(state))
        for dim in range(self.data_object.dimensions):
            l_bound = self.data_object.lowerBoundArray[dim]
            u_bound = self.data_object.upperBoundArray[dim]
            if state[dim] < l_bound:
                # print("******* Updated {} to {}".format(state[dim], l_bound + 0.000001))
                state[dim] = l_bound + 0.0000001
            elif state[dim] > u_bound:
                # print("******* Updated {} to {}".format(state[dim], u_bound - 0.000001))
                # x_val[dim] = 2 * u_bound - x_val[dim]
                state[dim] = u_bound - 0.0000001
        return state

    def evalModel(self, input=None, eval_var='v', model=None):
        output = None
        if eval_var is 'vp':
            x_v_t_pair = list(input[0])
            x_v_t_pair = x_v_t_pair + list(input[1])
            x_v_t_pair = x_v_t_pair + list(input[2])
            x_v_t_pair = x_v_t_pair + [input[4]]
            x_v_t_pair = np.asarray([x_v_t_pair], dtype=np.float64)
            predicted_vp = model.predict(x_v_t_pair)
            predicted_vp = predicted_vp.flatten()
            output = predicted_vp
            # print(predicted_vp)

        elif eval_var is 'v':
            xp_vp_t_pair = list(input[0])
            xp_vp_t_pair = xp_vp_t_pair + list(input[1])
            xp_vp_t_pair = xp_vp_t_pair + list(input[3])
            xp_vp_t_pair = xp_vp_t_pair + [input[4]]
            xp_vp_t_pair = np.asarray([xp_vp_t_pair], dtype=np.float64)
            predicted_v = model.predict(xp_vp_t_pair)
            predicted_v = predicted_v.flatten()
            output = predicted_v
            # print(predicted_v)

        return output

    def plotInvSenStaliroResults(self, trajectories, best_trajectory):

        u_x_min = self.usafelowerBoundArray[0]
        u_x_max = self.usafeupperBoundArray[0]
        u_y_min = self.usafelowerBoundArray[1]
        u_y_max = self.usafeupperBoundArray[1]

        u_verts = [
            (u_x_min, u_y_min),  # left, bottom
            (u_x_max, u_y_min),  # left, top
            (u_x_max, u_y_max),  # right, top
            (u_x_min, u_y_max),  # right, bottom
            (u_x_min, u_y_min),  # ignored
        ]

        i_x_min = self.data_object.lowerBoundArray[0]
        i_x_max = self.data_object.upperBoundArray[0]
        i_y_min = self.data_object.lowerBoundArray[1]
        i_y_max = self.data_object.upperBoundArray[1]

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

        u_patch = patches.PathPatch(u_path, facecolor='red', lw=2)
        ax.add_patch(u_patch)

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='none', lw=1)
        ax.add_patch(i_patch)

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]
        # colors = ['red', 'black', 'blue', 'brown', 'green']

        # dist_dict = {
        #         "0": "0-0.1",
        #         "1": "0.1-0.2",
        #         "2": "0.2-0.3",
        #         "3": "0.3-0.4",
        #         "4": "0.4-0.5",
        #         "5": "0.5-0.6",
        #         "6": "0.6-0.7",
        #         "7": "0.7-0.8",
        #         "8": "0.8-0.9",
        #         "9": ">=0.9"
        # }
        # ax.scatter(point[0], point[1], color=colors[idx], label=dist_dict.get(str(idx)))

        x_index = 0
        y_index = 1
        n_trajectories = len(trajectories)
        ax.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7], label='Reference Trajectory')

        for idx in range(1, n_trajectories - 1):
            pred_init = trajectories[idx][0]
            ax.scatter(pred_init[x_index], pred_init[y_index], color='g')

        ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], color=colors[1], label='Final Trajectory')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        # ax.set_xlim(5, 10)
        # ax.set_ylim(5, 30)
        plt.legend()

        plt.show()

    def plotInvSenResultsv1(self, trajectories, destinations, d_time_step, dimensions, best_trajectory):
        n_trajectories = len(trajectories)
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 10)]

        destination = destinations[0]
        if dimensions == 2:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7],
                     label='Reference Trajectory')
            # starting_state = trajectories[0][d_time_step]
            for idx in range(1, n_trajectories-1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b',
                     label='Final Trajectory')

            plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

            for destination in destinations:
                plt.plot(destination[x_index], destination[y_index], 'ko')

            # c_center_x = 1.75
            # c_center_y = -0.25
            # c_size = 0.1
            # deg = list(range(0, 360, 5))
            # deg.append(0)
            # xl = [c_center_x + c_size * math.cos(np.deg2rad(d)) for d in deg]
            # yl = [c_center_y + c_size * math.sin(np.deg2rad(d)) for d in deg]
            # plt.plot(xl, yl, color='k')
            plt.legend()
            plt.show()
        elif dimensions == 3:
            x_index = 0
            y_index = 1
            z_index = 2
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel('x')
            ax.set_xlabel('y')
            ax.set_xlabel('z')
            ax.plot3D(trajectories[0][:, x_index], trajectories[0][:, y_index], trajectories[0][:, z_index],
                      color=colors[7], label='Reference Trajectory')
            for idx in range(1, n_trajectories - 1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                ax.scatter3D(pred_init[x_index], pred_init[y_index], pred_init[z_index], color='green')
                ax.scatter3D(pred_destination[x_index], pred_destination[y_index], pred_destination[z_index],
                             color='red')

            ax.plot3D(trajectories[n_trajectories - 1][:, x_index], trajectories[n_trajectories - 1][:, y_index],
                      trajectories[n_trajectories-1][:, z_index], color='blue', label='Final Trajectory')

            ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black',
                         label='Destination z')

            for destination in destinations:
                ax.scatter3D(destination[x_index], destination[y_index], destination[z_index], color='black')
            plt.legend()
            plt.show()
        elif dimensions == 4:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7],
                     label='Reference Trajectory')
            # starting_state = trajectories[0][d_time_step]
            for idx in range(1, n_trajectories - 1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b',
                     label='Final Trajectory')

            plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

            for destination in destinations:
                plt.plot(destination[x_index], destination[y_index], 'ko')
            plt.legend()
            plt.show()
            plt.figure(1)
            x_index = 2
            y_index = 3
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], color=colors[7],
                     label='Reference Trajectory')
            # starting_state = trajectories[0][d_time_step]
            for idx in range(1, n_trajectories - 1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b',
                     label='Final Trajectory')

            plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

            for destination in destinations:
                plt.plot(destination[x_index], destination[y_index], 'ko')
            plt.legend()
            plt.show()
        elif dimensions == 6:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], 'b', label='Reference')
            for idx in range(1, n_trajectories-1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(trajectories[n_trajectories-1][:, x_index], trajectories[n_trajectories-1][:, y_index], 'g',
                     label='Final')
            plt.plot(destination[x_index], destination[y_index], 'y*', label='Actual destination')
            plt.legend()
            plt.show()

            plt.figure(1)
            x_index = 2
            y_index = 3
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], 'b', label='Reference')
            for idx in range(1, n_trajectories-1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(trajectories[n_trajectories-1][:, x_index], trajectories[n_trajectories-1][:, y_index], 'g',
                     label='Final')
            plt.plot(destination[x_index], destination[y_index], 'y*', label='Actual destination')
            plt.legend()
            plt.show()

            plt.figure(1)
            x_index = 4
            y_index = 5
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            plt.plot(trajectories[0][:, x_index], trajectories[0][:, y_index], 'b', label='Reference')
            for idx in range(1, n_trajectories-1):
                pred_init = trajectories[idx][0]
                pred_destination = trajectories[idx][d_time_step]
                plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')

            plt.plot(trajectories[n_trajectories-1][:, x_index], trajectories[n_trajectories-1][:, y_index], 'g',
                     label='Final')
            plt.plot(destination[x_index], destination[y_index], 'y*', label='Actual destination')
            plt.legend()
            plt.show()

    @staticmethod
    # For rrt algorithms repo based path. Plot them in html
    def plotInvSenResultsv2(self, trajectories, destination, d_time_step, dimensions, rand_area, path_idx, dynamics):

        ss_dimensions = []
        for idx in range(dimensions):
            tuple_idx = (rand_area[0][idx], rand_area[1][idx])
            ss_dimensions.append(tuple_idx)

        ss_dimensions = np.array(ss_dimensions)  # dimensions of Search Space
        # X_dimensions = np.array([(-2, 0), (0, 2), (0, 1)])
        # create search space
        search_space = SearchSpace(ss_dimensions)

        n_trajectories = len(trajectories)

        plot = Plot(self.eval_dir+"/rrtfigs/", "rrt_" + dynamics + "_" + str(dimensions) + "d_" + str(path_idx))
        # plot.plot_tree(X, rrt.trees)
        ref_traj = trajectories[0]
        ref_traj_tupled = tuple(map(tuple, ref_traj))

        plot.plot_path(search_space, ref_traj_tupled, "blue")

        final_traj = trajectories[n_trajectories-1]
        final_traj_tupled = tuple(map(tuple, final_traj))

        plot.plot_path(search_space, final_traj_tupled, "green")

        for idx in range(1, n_trajectories - 1):
            pred_init = trajectories[idx][0]
            pred_destination = trajectories[idx][d_time_step]

            pred_init_tuple = tuple(pred_init)
            pred_dest_tuple = tuple(pred_destination)
            plot.plot_start(search_space, pred_init_tuple, color="green")
            plot.plot_goal(search_space, pred_dest_tuple, color="red")

        plot.plot_goal(search_space, tuple(destination), color="orange")

        # if path is not None:
        #     plot.plot_path(searchSpace, rrt_path_points)
        # plot.plot_obstacles(X, Obstacles)

        plot.draw(auto_open=False)

    def reachDestInvRRTPaths(self, dests=None, d_time_steps=None, threshold=0.01, correction_steps=[50],
                          scaling_factors=[0.01], i_state=None, dnn_or_rbf='RBF', layers=1, neurons=256,
                              act_fn='ReLU', adaptation_run=False, rand_area=None, n_paths=2):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]
        d_time_step = d_time_steps[0]

        assert self.data_object is not None

        dimensions = self.data_object.getDimensions()
        self.rrt_run = True

        model_rbf_f_name = self.eval_dir + 'models/model_vp_2_v_'
        model_rbf_f_name = model_rbf_f_name + self.data_object.dynamics
        model_rbf_f_name = model_rbf_f_name + "_" + dnn_or_rbf
        model_rbf_f_name = model_rbf_f_name + "_" + str(layers)
        model_rbf_f_name = model_rbf_f_name + "_" + str(neurons)
        model_rbf_f_name = model_rbf_f_name + "_" + act_fn
        if supp is True:
            model_rbf_f_name = model_rbf_f_name + "_" + "supp"
        model_rbf_f_name = model_rbf_f_name + '.h5'

        if path.exists(model_rbf_f_name):
            if dnn_or_rbf is 'dnn':
                model_rbf_v = load_model(model_rbf_f_name, compile=True)
            else:
                model_rbf_v = load_model(model_rbf_f_name, compile=True, custom_objects={'RBFLayer': RBFLayer})
        else:
            print("Model file " + model_rbf_f_name + " does not exists.")
            return
        print("Model file " + model_rbf_f_name)

        paths_list = []

        for p_idx in range(n_paths):
            rrt_path_points = []
            obstacleList = [
                (1.75, -0.25, 0.1)
            ]  # [x, y, radius]
            if dimensions == 2:
                rrt = RRTV1(start=ref_traj[d_time_step], goal=dest, rand_area=rand_area, obstacle_list=obstacleList)
                rrt_path_points = rrt.planning(animation=False)

                rrt_path_points = rrt_path_points[::-1]
            elif dimensions == 3:
                ss_dimensions = []
                for idx in range(dimensions):
                    tuple_idx = (rand_area[0][idx], rand_area[1][idx])
                    ss_dimensions.append(tuple_idx)

                ss_dimensions = np.array(ss_dimensions)  # dimensions of Search Space
                # X_dimensions = np.array([(-2, 0), (0, 2), (0, 1)])
                # create search space
                search_space = SearchSpace(ss_dimensions)

                # # obstacles
                # Obstacles = np.array([])
                print(ss_dimensions)

                # uncomment this if you want to use it for 2-d systems
                # x_init = (ref_traj[d_time_step][0], ref_traj[d_time_step][1])  # starting location
                # x_goal = (dest[0], dest[1])  # goal location

                x_init = (ref_traj[d_time_step][0], ref_traj[d_time_step][1], ref_traj[d_time_step][2])
                # starting location
                x_goal = (dest[0], dest[1], dest[2])  # goal location

                print(x_init, x_goal)
                Q = np.array([(8, 2)])  # length of tree edges
                r = 5  # length of smallest edge to check for intersection with obstacles
                max_samples = 100  # max number of samples to take before timing out
                prc = 0.01  # probability of checking for a connection to goal

                # create rrt_search
                rrt = RRT(search_space, Q, x_init, x_goal, max_samples, r, prc)
                rrt_path_points_tuples = rrt.rrt_search()
                print(rrt_path_points_tuples)
                rrt_path_points = []
                for point in rrt_path_points_tuples:
                    path_point = [point[0], point[1], point[2]]
                    rrt_path_points.append(path_point)

            # print(rrt_paths)

            rrt_path = []
            for idx in range(len(rrt_path_points)-1):
                rrt_segment = [np.array(rrt_path_points[idx]), np.array(rrt_path_points[idx+1])]
                rrt_path.append(rrt_segment)

            paths_list.append(rrt_path)
            print(rrt_path)

        # print(paths_list)

        adaptation_factor = 3/4

        for paths in paths_list:
            # print(paths)
            for s_factor in scaling_factors:

                for steps in correction_steps:
                    if steps == 1:
                        start_time = time.time()
                        self.reachDestInvBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                  threshold=threshold, model_v=model_rbf_v, iter_bound=self.iter_count,
                                                  scaling_factor=s_factor, adaptation_run=adaptation_run,
                                                  adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvNonBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold, model_v=model_rbf_v, correction_steps=steps,
                                                     scaling_factor=s_factor, iter_bound=self.iter_count,
                                                     adaptation_run=adaptation_run, adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))

    def reachDestInvPaths(self, dests=None, d_time_steps=None, threshold=0.01, correction_steps=[50],
                          scaling_factors=[0.01], i_state=None, dnn_or_rbf='RBF', layers=1, neurons=256, act_fn='ReLU',
                          adaptation_run=False):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]
        d_time_step = d_time_steps[0]

        assert self.data_object is not None

        model_rbf_f_name = self.eval_dir + 'models/model_vp_2_v_'
        model_rbf_f_name = model_rbf_f_name + self.data_object.dynamics
        model_rbf_f_name = model_rbf_f_name + "_" + dnn_or_rbf
        model_rbf_f_name = model_rbf_f_name + "_" + str(layers)
        model_rbf_f_name = model_rbf_f_name + "_" + str(neurons)
        model_rbf_f_name = model_rbf_f_name + "_" + act_fn
        model_rbf_f_name = model_rbf_f_name + '.h5'

        if path.exists(model_rbf_f_name):
            if dnn_or_rbf is 'dnn':
                model_rbf_v = load_model(model_rbf_f_name, compile=True)
            else:
                model_rbf_v = load_model(model_rbf_f_name, compile=True, custom_objects={'RBFLayer': RBFLayer})
        else:
            print("Model file " + model_rbf_f_name + " does not exists.")
            return
        print("Model file " + model_rbf_f_name)

        # original_vector = dest - ref_traj[d_time_step]
        # original_vector_div = original_vector/3
        # path1_src = ref_traj[d_time_step]
        # path1_dest = ref_traj[d_time_step] + original_vector_div
        # path1 = [path1_src, path1_dest]
        # path2_dest = path1_dest + original_vector_div
        # path2 = [path1_dest, path2_dest]
        # path3 = [path2_dest, dest]
        # # paths_list = [[path1, path2, path3]]
        paths_list = [[[ref_traj[d_time_step], dest]]]

        # Divide the straight line path in 3 and rotate the first segment by some degrees to give an impression that
        # this is an RRT-like path
        # angle15 = np.pi/12
        # rot_matrix15 = np.array([[math.cos(angle15), math.sin(angle15)], [-math.sin(angle15), math.cos(angle15)]])
        # original_vector_div_rot15 = np.dot(rot_matrix15, original_vector_div)
        #
        # path1_dest = ref_traj[d_time_step] + original_vector_div_rot15
        # path2_dest = path1_dest + original_vector_div
        # path1 = [path1_src, path1_dest]
        # path2 = [path1_dest, path2_dest]
        # path3 = [path2_dest, dest]
        # paths_list.append([path1, path2, path3])
        #
        # angle345 = 2*np.pi - np.pi/12
        # rot_matrix345 = np.array([[math.cos(angle345), math.sin(angle345)],
        # [-math.sin(angle345), math.cos(angle345)]])
        # original_vector_div_rot345 = np.dot(rot_matrix345, original_vector_div)
        #
        # path1_dest = ref_traj[d_time_step] + original_vector_div_rot345
        # path2_dest = path1_dest + original_vector_div
        # path1 = [path1_src, path1_dest]
        # path2 = [path1_dest, path2_dest]
        # path3 = [path2_dest, dest]
        # paths_list.append([path1, path2, path3])

        # print(paths_list)

        f_iterations = None
        min_dist = None
        adaptation_factor = 1/2
        rel_dist = None

        for paths in paths_list:
            # print(paths)
            for s_factor in scaling_factors:

                for steps in correction_steps:
                    if steps == 1:
                        print(" *** Baseline Greedy *** \n")
                        start_time = time.time()
                        self.reachDestInvBaselineGreedy(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                        threshold=threshold, model_v=model_rbf_v,
                                                        iter_bound=self.iter_count,
                                                        scaling_factor=s_factor, adaptation_run=adaptation_run,
                                                        adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))
                        start_time = time.time()
                        self.reachDestInvBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                  threshold=threshold, model_v=model_rbf_v, iter_bound=self.iter_count,
                                                  scaling_factor=s_factor, adaptation_run=adaptation_run,
                                                  adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvNonBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold,  model_v=model_rbf_v, correction_steps=steps,
                                                     scaling_factor=s_factor, iter_bound=self.iter_count,
                                                     adaptation_run=adaptation_run, adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))

        if self.staliro_run is True:
            return f_iterations, min_dist, rel_dist

    def reachDestInvBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, iter_bound, scaling_factor,
                             adaptation_factor, adaptation_run, rand_area=None, dynamics=None):

        original_scaling_factor = scaling_factor
        dimensions = self.data_object.getDimensions()
        n_paths = len(paths)
        x_val = ref_traj[0]
        # xp_val = paths[0][0]
        # print(x_val, xp_val)
        trajectories = [ref_traj]
        rrt_dests = []

        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            scaling_factor = original_scaling_factor
            print("***** path idx " + str(path_idx) + " s_factor " + str(scaling_factor) + " correction steps 1")
            # print("destination " + str(dest))
            # print(ref_traj[0])
            before_adaptation_iter = 0
            after_adaptation_iter = 0
            adaptation_applied = False
            prev_dist_while_adaptation = None
            v_val = dest - x_val
            print(xp_val, dest)
            vp_val = dest - xp_val
            vp_norm = norm(vp_val, 2)
            min_iter = 0
            t_val = d_time_step
            vp_val = [val / vp_norm for val in vp_val]  # Normalized
            print(iter_bound)
            dist = vp_norm
            print("Starting distance: " + str(dist))
            final_dist = dist
            original_distance = final_dist
            min_dist = final_dist
            best_state = x_val
            adaptations = 0
            halt_adaptation = False
            while final_dist > threshold and (before_adaptation_iter + after_adaptation_iter) < iter_bound:
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                v_val = predicted_v_scaled
                vp_val = dest - xp_val
                vp_norm = norm(vp_val, 2)
                dist = vp_norm
                vp_val = [val / vp_norm for val in vp_val]  # Normalized
                t_val = d_time_step

                if adaptation_applied is False:
                    before_adaptation_iter = before_adaptation_iter + 1
                else:
                    after_adaptation_iter = after_adaptation_iter + 1

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    best_state = x_val
                    if adaptation_applied is False:
                        min_iter = before_adaptation_iter
                    else:
                        min_iter = before_adaptation_iter + after_adaptation_iter

                if adaptation_run is True and final_dist is not None and (final_dist - dist > 1e-7) \
                        and halt_adaptation is False:
                    # print("Changing the scaling factor")
                    scaling_factor += 0.0002

                elif adaptation_run is True and dist >= final_dist and halt_adaptation is False:
                    current_rel_dist = final_dist / original_distance
                    if adaptations == 2 or (prev_dist_while_adaptation is not None and
                                            current_rel_dist > prev_dist_while_adaptation):
                        # trajectories.pop()
                        print("Adaptations reached it's limits")
                        halt_adaptation = True
                        # if self.staliro_run is True:
                        #     break
                    else:
                        print("Before adaptation relative distance " + str(current_rel_dist))
                        # print("*** distance increased ****")
                        if prev_dist_while_adaptation is None:
                            prev_dist_while_adaptation = current_rel_dist
                        adaptation_applied = True
                        scaling_factor = scaling_factor * adaptation_factor
                        adaptations += 1

                final_dist = dist
                # scaling_factor = scaling_factor + orig_scaling_factor/10
            best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]
            print("Final relative distance " + str(final_dist/original_distance))
            print("Min relative distance " + str(min_dist/original_distance))
            # print("Final dist: " + str(final_dist))
            # print("Min dist: " + str(min_dist))
            print("Min iter: " + str(min_iter))
            # print("Terminating state: " + str(x_val))
            print("Before adaptation iterations: " + str(before_adaptation_iter))
            print("Final iterations: " + str(before_adaptation_iter + after_adaptation_iter))

            self.f_iterations = before_adaptation_iter + after_adaptation_iter
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

            self.plotInvSenResultsv1(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResultsv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx,dynamics)

    def reachDestInvNonBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, correction_steps, iter_bound,
                                scaling_factor, adaptation_factor, adaptation_run, rand_area=None, dynamics=None):

        original_scaling_factor = scaling_factor
        dimensions = self.data_object.getDimensions()
        n_paths = len(paths)
        x_val = ref_traj[0]
        # xp_vals_list = []
        # x_vals_list = []
        # xp_val = paths[0][0]
        # print(x_val, xp_val)
        trajectories = [ref_traj]
        rrt_dests = []
        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            scaling_factor = original_scaling_factor
            print("***** path idx " + str(path_idx) + " s_factor " + str(scaling_factor) + " correction steps " +
                  str(correction_steps))
            before_adaptation_iter = 0
            after_adaptation_iter = 0
            adaptation_applied = False
            prev_dist_while_adaptation = None
            adaptations = 0
            x_vals = []
            xp_vals = []
            v_val = dest - x_val
            vp_val = dest - xp_val
            vp_norm = norm(vp_val, 2)
            dist = vp_norm
            original_distance = dist
            final_dist = dist
            print("Starting distance: " + str(dist))
            min_dist = dist
            best_state = x_val
            halt_adaptation = False
            min_iter = 0

            while final_dist > threshold and (before_adaptation_iter + after_adaptation_iter) < iter_bound:

                x_vals.append(x_val)
                xp_vals.append(xp_val)
                t_val = d_time_step
                vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
                vp_val_scaled = [val * scaling_factor for val in vp_val_normalized]
                step = 0
                prev_pred_dist = None

                while step < correction_steps:
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)
                    predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                    predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                    new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                    x_val = new_init_state[0]
                    xp_val = xp_val + vp_val_scaled
                    v_val = predicted_v_scaled
                    vp_val = dest - xp_val
                    vp_norm = norm(vp_val, 2)
                    x_vals.append(x_val)
                    xp_vals.append(xp_val)
                    pred_dist = vp_norm
                    vp_val_normalized = [val/vp_norm for val in vp_val]
                    # print("new distance: " + str(dist))
                    t_val = d_time_step
                    step += 1
                    if prev_pred_dist is not None and prev_pred_dist < pred_dist:
                        x_vals.pop()
                        xp_vals.pop()
                        break
                    prev_pred_dist = pred_dist

                new_traj = self.data_object.generateTrajectories(r_states=[x_vals[len(x_vals) - 1]])[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                vp_val = dest - xp_val
                vp_norm = norm(vp_val, 2)
                dist = vp_norm

                if dist < min_dist:
                    min_dist = dist
                    best_state = x_val
                    if adaptation_applied is False:
                        min_iter = before_adaptation_iter
                    else:
                        min_iter = before_adaptation_iter + after_adaptation_iter

                if adaptation_applied is False:
                    before_adaptation_iter = before_adaptation_iter + 1
                else:
                    after_adaptation_iter = after_adaptation_iter + 1

                trajectories.append(new_traj)

                if adaptation_run is True and final_dist is not None and (final_dist - dist > 1e-6) and \
                        halt_adaptation is False:
                    # print("Changing the scaling factor")
                    scaling_factor += 0.0002

                elif adaptation_run is True and dist >= final_dist and halt_adaptation is False:
                    current_rel_dist = final_dist / original_distance
                    if adaptations == 2 or (prev_dist_while_adaptation is not None and
                                            current_rel_dist > prev_dist_while_adaptation):
                        # trajectories.pop()
                        print("Adaptations reached it's limits")
                        halt_adaptation = True
                        # if self.staliro_run is True:
                        #     break
                    else:
                        print("Before adaptation relative distance " + str(current_rel_dist))
                        # print("*** distance increased ****")
                        if prev_dist_while_adaptation is None:
                            prev_dist_while_adaptation = current_rel_dist
                        adaptation_applied = True
                        scaling_factor = scaling_factor * adaptation_factor
                        adaptations += 1

                final_dist = dist
            min_rel_dist = min_dist / original_distance
            best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]
            print("Final relative distance " + str(final_dist / original_distance))
            print("Min relative distance: " + str(min_rel_dist))
            print("Min iter: " + str(min_iter))
            print("Final dist: " + str(final_dist))
            # print("Terminating state: " + str(x_val))
            print("Before adaptation iterations: " + str(before_adaptation_iter))
            print("Final iterations: " + str(before_adaptation_iter + after_adaptation_iter))
            self.f_iterations = before_adaptation_iter + after_adaptation_iter
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

            self.plotInvSenResultsv1(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResultsv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx, dynamics)

            # if self.staliro_run:
            #     self.plotInvSenStaliroResults(trajectories, best_trajectory)
            # return before_adaptation_iter + after_adaptation_iter, min_dist, min_rel_dist

    def get_vp_direction_greedy(self, dimensions, scaling_factor, dest, xp_val):
        i_matrix = np.eye(dimensions)
        vp_direction_dist = None
        vp_direction = None
        for i_m_idx in range(dimensions):

            vec_1 = i_matrix[i_m_idx]
            vec_1 = [val * scaling_factor for val in vec_1]
            temp_xp_val = xp_val + vec_1
            temp_dist = norm(dest - temp_xp_val, 2)
            if vp_direction_dist is None or temp_dist < vp_direction_dist:
                vp_direction_dist = temp_dist
                vp_direction = vec_1

            vec_2 = i_matrix[i_m_idx]
            vec_2 = [val * -scaling_factor for val in vec_2]
            temp_xp_val = xp_val + vec_2
            temp_dist = norm(dest - temp_xp_val, 2)
            if temp_dist < vp_direction_dist:
                vp_direction_dist = temp_dist
                vp_direction = vec_2

        return vp_direction

    def reachDestInvBaselineGreedy(self, ref_traj, paths, d_time_step, threshold, model_v, iter_bound, scaling_factor,
                                   adaptation_factor, adaptation_run):

        original_scaling_factor = scaling_factor
        dimensions = self.data_object.getDimensions()
        n_paths = len(paths)
        x_val = ref_traj[0]
        # xp_val = paths[0][0]
        # print(x_val, xp_val)
        trajectories = [ref_traj]
        rrt_dests = []
        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            scaling_factor = original_scaling_factor
            print("***** path idx " + str(path_idx) + " s_factor " + str(scaling_factor) + " correction steps 1")
            # print("destination " + str(dest))
            # print(ref_traj[0])
            before_adaptation_iter = 0
            after_adaptation_iter = 0
            adaptation_applied = False
            prev_dist_while_adaptation = None
            v_val = dest - x_val
            print(xp_val, dest)

            # vp_val = dest - xp_val
            vp_direction = self.get_vp_direction_greedy(dimensions, scaling_factor, dest, xp_val)
            vp_val = vp_direction
            print("vp direction " + str(vp_direction))
            vp_norm = norm(vp_val, 2)
            min_iter = before_adaptation_iter + after_adaptation_iter
            t_val = d_time_step
            vp_val = [val / vp_norm for val in vp_val]  # Normalized
            print(iter_bound)
            # dist = vp_norm
            final_dist = norm(dest-xp_val, 2)
            # final_dist = dist
            print("Starting distance: " + str(final_dist))
            original_distance = final_dist
            min_dist = final_dist
            best_state = x_val
            halt_adaptation = False
            adaptations = 0
            while final_dist > threshold and (before_adaptation_iter + after_adaptation_iter) < iter_bound:
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]
                v_val = predicted_v_scaled
                # vp_val = dest - xp_val
                # print("vp direction " + str(vp_direction))
                vp_direction = self.get_vp_direction_greedy(dimensions, scaling_factor, dest, xp_val)
                vp_val = vp_direction
                vp_norm = norm(vp_val, 2)
                dist = norm(dest - xp_val, 2)
                vp_val = [val / vp_norm for val in vp_val]  # Normalized
                t_val = d_time_step

                if adaptation_applied is False:
                    before_adaptation_iter = before_adaptation_iter + 1
                else:
                    after_adaptation_iter = after_adaptation_iter + 1

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    best_state = x_val
                    if adaptation_applied is False:
                        min_iter = before_adaptation_iter
                    else:
                        min_iter = before_adaptation_iter + after_adaptation_iter

                if adaptation_run is True and final_dist is not None and (final_dist - dist > 1e-7) \
                        and halt_adaptation is False:
                    # print("Changing the scaling factor")
                    scaling_factor += 0.0002

                elif adaptation_run is True and dist >= final_dist and halt_adaptation is False:
                    current_rel_dist = final_dist / original_distance
                    if adaptations == 2 or (prev_dist_while_adaptation is not None and
                                            current_rel_dist > prev_dist_while_adaptation):
                        # trajectories.pop()
                        print("Adaptations reached it's limits")
                        halt_adaptation = True
                        # if self.staliro_run is True:
                        #     break
                    else:
                        print("Before adaptation relative distance " + str(current_rel_dist))
                        # print("*** distance increased ****")
                        if prev_dist_while_adaptation is None:
                            prev_dist_while_adaptation = current_rel_dist
                        adaptation_applied = True
                        scaling_factor = scaling_factor * adaptation_factor
                        adaptations += 1

                final_dist = dist
                # scaling_factor = scaling_factor + orig_scaling_factor/10
            best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]
            print("Final relative distance " + str(final_dist/original_distance))
            print("Min relative distance " + str(min_dist/original_distance))
            print("Final dist: " + str(final_dist))
            print("Min dist: " + str(min_dist))
            print("Min iter: " + str(min_iter))
            print("Terminating state: " + str(x_val))
            print("Before adaptation iterations: " + str(before_adaptation_iter))
            print("Final iterations: " + str(before_adaptation_iter + after_adaptation_iter))
            self.plotInvSenResultsv1(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResultsv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx,dynamics)
            self.f_iterations = before_adaptation_iter + after_adaptation_iter
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

    def plotFwdSenResults(self, ref_traj, delta_vecs, actual_dests, pred_dests):
        if self.data_object.dimensions == 2:
            plt.figure(1)
            x_index = 0
            y_index = 1
            plt.xlabel('x' + str(x_index))
            plt.ylabel('x' + str(y_index))
            x_val = ref_traj[0]
            plt.plot(ref_traj[:, x_index], ref_traj[:, y_index], 'b', label='Reference')
            for idx in range(len(delta_vecs)):
                delta_vec = delta_vecs[idx]
                neighbor_state = [x_val[i] + delta_vec[i] for i in range(len(delta_vec))]
                plt.plot(neighbor_state[x_index], neighbor_state[y_index], 'g*')

            for idx in range(len(actual_dests)):
                actual_dest = actual_dests[idx]
                pred_dest = pred_dests[idx]
                plt.plot(pred_dest[x_index], pred_dest[y_index], 'g^')
                plt.plot(actual_dest[x_index], actual_dest[y_index], 'r*')
            plt.legend()
            plt.show()

    def plotFwdsenTrajectories(self, ref_trajs, predicted_trajs, actual_trajs, max_time, x_vals=None, xp_vals=None):
        x_index = 0
        y_index = 1
        n_trajs = len(predicted_trajs)
        plt.plot(ref_trajs[0][0:max_time, x_index], ref_trajs[0][0:max_time, y_index], 'b', label='Reference trajectory')
        plt.plot(predicted_trajs[n_trajs-1][0:max_time, x_index], predicted_trajs[n_trajs-1][0:max_time, y_index], 'r',
                 label='Predicted trajectory')
        plt.plot(actual_trajs[n_trajs-1][0:max_time, x_index], actual_trajs[n_trajs-1][0:max_time, y_index], 'g',
                 label='Actual trajectory')
        # for idx in range(1, n_trajs):
        #     ref_traj = ref_trajs[idx]
        #     plt.plot(ref_traj[0:max_time, x_index], ref_traj[0:max_time, y_index], 'b')
        #     predicted_traj = predicted_trajs[idx]
        #     actual_traj = actual_trajs[idx]
        #     plt.plot(predicted_traj[0:max_time, x_index], predicted_traj[0:max_time, y_index], 'r')
        #     plt.plot(actual_traj[0:max_time, x_index], actual_traj[0:max_time, y_index], 'g')

        if x_vals is not None and len(x_vals) > 0:
            plt.plot(x_vals[0][0], x_vals[0][1], 'r*', label='Initial states')
            plt.plot(xp_vals[0][0], xp_vals[0][1], 'g*', label='Pred destination states')
            for idx in range(1, len(x_vals)):
                plt.plot(x_vals[idx][0], x_vals[idx][1], 'r*')
                plt.plot(xp_vals[idx][0], xp_vals[idx][1], 'g*')

        plt.legend()
        plt.show()

    def spaceExploreFwd(self, scaling_factors=0.01, dnn_or_rbf='RBF', layers=1, neurons=256,
                        act_fn='ReLU', d_time_step=10):

        n_vectors = 2
        max_time = 200
        assert self.data_object is not None

        model_rbf_f_name = self.eval_dir + 'models/model_v_2_vp_'
        model_rbf_f_name = model_rbf_f_name + self.data_object.dynamics
        model_rbf_f_name = model_rbf_f_name + "_" + dnn_or_rbf
        model_rbf_f_name = model_rbf_f_name + "_" + str(layers)
        model_rbf_f_name = model_rbf_f_name + "_" + str(neurons)
        model_rbf_f_name = model_rbf_f_name + "_" + act_fn
        model_rbf_f_name = model_rbf_f_name + '.h5'

        if path.exists(model_rbf_f_name):
            if dnn_or_rbf is 'dnn':
                model_rbf_vp = load_model(model_rbf_f_name, compile=True)
            else:
                model_rbf_vp = load_model(model_rbf_f_name, compile=True, custom_objects={'RBFLayer': RBFLayer})
        else:
            print("Model file " + model_rbf_f_name + " does not exists.")
            return
        print("Model file " + model_rbf_f_name)

        vecs_in_unit_circle = generate_points_in_circle(n_samples=n_vectors, dim=self.data_object.dimensions)

        delta_vecs = []
        for vec in vecs_in_unit_circle:
            delta_vec = [val * scaling_factors for val in vec]
            delta_vecs.append(delta_vec)

        pred_trajs = []
        actual_trajs = []
        original_ref_trajs = []
        for idx in range(n_vectors):
            ref_traj = self.data_object.generateTrajectories(samples=1)[0]
            original_ref_trajs.append(ref_traj)
            plt.plot(ref_traj[0:max_time, 0], ref_traj[0:max_time, 1], 'b')
        plt.show()

        xp_vals = []
        x_vals = []
        for idx in range(n_vectors):
            d_time_step = random.randint(50, 150)
            ref_traj = original_ref_trajs[idx]
            delta_vec = delta_vecs[idx]
            # pred_traj = None
            x_val = ref_traj[0]
            x_vals.append(x_val)
            xp_vals.append(ref_traj[d_time_step])
            for idy in range(0, 30):
                v_val = delta_vec
                v_norm = norm(v_val, 2)
                v_val_scaled = [val / v_norm for val in v_val]
                neighbor_state = [x_val[i] + delta_vec[i] for i in range(len(delta_vec))]
                pred_traj = [neighbor_state]
                for t_step in range(1, max_time):
                    xp_val = ref_traj[t_step]
                    t_val = t_step
                    vp_val = v_val
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val_scaled, vp_val, t_val)
                    predicted_vp = self.evalModel(input=data_point, eval_var='vp', model=model_rbf_vp)
                    predicted_vp = predicted_vp * v_norm
                    pred_dest = [xp_val[i] + predicted_vp[i] for i in range(len(xp_val))]
                    pred_traj.append(pred_dest)
                ref_traj = pred_traj
                x_val = neighbor_state
                x_vals.append(x_val)
                xp_vals.append(ref_traj[d_time_step])

            print("Done****")
            ref_traj = np.array(ref_traj)
            pred_trajs.append(ref_traj)
            actual_traj = self.data_object.generateTrajectories(r_states=[ref_traj[0]])[0]
            actual_trajs.append(actual_traj)
        self.plotFwdsenTrajectories(original_ref_trajs, pred_trajs, actual_trajs, max_time, x_vals, xp_vals)

    def spaceExploreFwd2(self, scaling_factors=0.01, dnn_or_rbf='RBF', layers=1, neurons=256,
                        act_fn='ReLU'):

        n_vectors = 4
        max_time = 200
        assert self.data_object is not None

        model_rbf_f_name = self.eval_dir + 'models/model_v_2_vp_'
        model_rbf_f_name = model_rbf_f_name + self.data_object.dynamics
        model_rbf_f_name = model_rbf_f_name + "_" + dnn_or_rbf
        model_rbf_f_name = model_rbf_f_name + "_" + str(layers)
        model_rbf_f_name = model_rbf_f_name + "_" + str(neurons)
        model_rbf_f_name = model_rbf_f_name + "_" + act_fn
        model_rbf_f_name = model_rbf_f_name + '.h5'

        if path.exists(model_rbf_f_name):
            if dnn_or_rbf is 'dnn':
                model_rbf_vp = load_model(model_rbf_f_name, compile=True)
            else:
                model_rbf_vp = load_model(model_rbf_f_name, compile=True, custom_objects={'RBFLayer': RBFLayer})
        else:
            print("Model file " + model_rbf_f_name + " does not exists.")
            return
        print("Model file " + model_rbf_f_name)

        vecs_in_unit_circle = generate_points_in_circle(n_samples=10, dim=self.data_object.dimensions)

        delta_vecs = []
        for vec in vecs_in_unit_circle:
            delta_vec = [val * scaling_factors for val in vec]
            delta_vecs.append(delta_vec)

        pred_trajs = []
        actual_trajs = []
        original_ref_trajs = []
        for idx in range(n_vectors):
            ref_traj = self.data_object.generateTrajectories(samples=1)[0]
            original_ref_trajs.append(ref_traj)

        xp_vals = []
        x_vals = []
        d_time_step = random.randint(20, 100)
        ref_traj = original_ref_trajs[0]
        plt.plot(ref_traj[0:max_time, 0], ref_traj[0:max_time, 1], 'b')
        plt.show()
        original_ref_trajs = [ref_traj]
        for idx in range(n_vectors):
            # ref_traj = original_ref_trajs[idx]
            delta_vec = delta_vecs[random.randint(0, 9)]
            x_val = ref_traj[0]
            x_vals.append(x_val)
            xp_vals.append(ref_traj[d_time_step])
            for idy in range(0, 20):
                v_val = delta_vec
                v_norm = norm(v_val, 2)
                v_val_scaled = [val / v_norm for val in v_val]
                neighbor_state = [x_val[i] + delta_vec[i] for i in range(len(delta_vec))]
                pred_traj = [neighbor_state]
                for t_step in range(1, max_time):
                    xp_val = ref_traj[t_step]
                    t_val = t_step
                    vp_val = v_val
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val_scaled, vp_val, t_val)
                    predicted_vp = self.evalModel(input=data_point, eval_var='vp', model=model_rbf_vp)
                    predicted_vp = predicted_vp * v_norm
                    pred_dest = [xp_val[i] + predicted_vp[i] for i in range(len(xp_val))]
                    pred_traj.append(pred_dest)
                ref_traj = pred_traj
                x_val = neighbor_state
                x_vals.append(x_val)
                xp_vals.append(ref_traj[d_time_step])

            print("Done**** " + str(idx))
            ref_traj = np.array(ref_traj)
            pred_trajs.append(ref_traj)
            actual_traj = self.data_object.generateTrajectories(r_states=[ref_traj[0]])[0]
            actual_trajs.append(actual_traj)
            self.plotFwdsenTrajectories(original_ref_trajs, pred_trajs, actual_trajs, max_time, x_vals, xp_vals)



