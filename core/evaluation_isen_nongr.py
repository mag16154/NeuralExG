from evaluation import Evaluation
from sampler import generateRandomStates
from matplotlib.path import Path
import matplotlib.patches as patches
import time
import numpy as np
import matplotlib.pyplot as plt
from frechet import norm

# version 1 RRT
from rrtv1 import RRTV1

# version 2 RRT
import sys
sys.path.append('./rrtAlgorithms/src/')
from rrt.rrt import RRT
from search_space.search_space import SearchSpace
from utilities.plotting import Plot


class EvaluationInvSenNonGr(Evaluation):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU'):
        Evaluation.__init__(self, dynamics=dynamics, sensitivity='Inv', dnn_rbf=dnn_rbf, layers=layers,
                            neurons=neurons, act_fn=act_fn, grad_run=False)
        self.f_iterations = 0
        self.f_dist = None
        self.f_rel_dist = None
        self.staliro_run = False
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []

    def getFIterations(self):
        return self.f_iterations

    def getFDistance(self):
        return self.f_dist

    def getFRelDistance(self):
        return self.f_rel_dist

    def setStaliroRun(self):
        self.staliro_run = True

    def setUnsafeSet(self, lowerBound, upperBound):
        self.usafelowerBoundArray = lowerBound
        self.usafeupperBoundArray = upperBound
        self.staliro_run = True

    def generateRandomUnsafeStates(self, samples):
        states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        return states

    def plotInvSenResults(self, trajectories, destinations, d_time_step, dimensions, best_trajectory,
                            x_vals=None, xp_vals=None):
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
                else:
                    for idx in range(1, n_trajectories - 1):
                        # print("******* Here ")
                        trajectory = trajectories[idx]
                        pred_init = trajectory[0]
                        pred_destination = trajectory[d_time_step]
                        plt.plot(pred_init[x_index], pred_init[y_index], 'g^')
                        plt.plot(pred_destination[x_index], pred_destination[y_index], 'r^')
                        plt.plot(trajectory[:, x_index], trajectory[:, y_index], 'g')

                plt.plot(trajectories[1][:, x_index], trajectories[1][:, y_index], 'g', label='Intermediate trajectories')
                plt.plot(best_trajectory[:, x_index], best_trajectory[:, y_index], 'b', label='Final trajectory')

                plt.plot(destination[x_index], destination[y_index], 'ko', label='Destination z')

                for destination in destinations:
                    plt.plot(destination[x_index], destination[y_index], 'ko')
                plt.legend()
                plt.show()

    def reachDestInvPaths(self, dests=None, d_time_steps=None, threshold=0.01, i_state=None):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]
        if d_time_steps is not None:
            d_time_step = d_time_steps[0]
        else:
            min_dist = 200.0
            min_idx = 0
            for idx in range(len(ref_traj)):
                curr_dist = norm(dest - ref_traj[idx], 2)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_idx = idx
            print(min_dist, min_idx)
            d_time_step = min_idx

        assert self.data_object is not None

        trained_model = self.getModel()
        if trained_model is None:
            return

        paths_list = [[[ref_traj[d_time_step], dest]]]

        f_iterations = None
        min_dist = None
        rel_dist = None

        for paths in paths_list:
            # print(paths)
            start_time = time.time()
            self.reachDestInvBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step, threshold=threshold,
                                      model_v=trained_model, iter_bound=self.iter_count, dynamics=self.dynamics)
            print("Time taken: " + str(time.time() - start_time))

        if self.staliro_run is True:
            return f_iterations, min_dist, rel_dist

    '''
    ReachDestination for correction period 1 (without greedy).
    It can be removed later as reachDestInvNonBaseline works just fine for correction period = 1 as well.
    '''
    def reachDestInvBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, iter_bound, dynamics=None):

        n_paths = len(paths)
        x_val = ref_traj[0]
        trajectories = [ref_traj]
        rrt_dests = []
        dimensions = self.data_object.getDimensions()

        for path_idx in range(n_paths):
            path = paths[path_idx]
            xp_val = path[0]
            dest = path[1]
            rrt_dests.append(dest)
            print("***** path idx " + str(path_idx) + " correction steps 1")
            v_val = dest - x_val
            vp_val = dest - xp_val
            min_iter = 0
            t_val = d_time_step
            dist = norm(vp_val, 2)
            print("Starting distance: " + str(dist))
            original_distance = dist
            min_dist = dist
            best_state = x_val
            iteration = 0

            while dist > threshold and iteration < iter_bound:
                if original_distance < 0.05:
                    break
                iteration = iteration + 1
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val, t_val)
                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                predicted_v = 4/5 * predicted_v
                new_init_state = [self.check_for_bounds(x_val + predicted_v)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]

                v_val = predicted_v
                vp_val = dest - xp_val
                dist = norm(vp_val, 2)
                t_val = d_time_step

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    best_state = x_val
                    min_iter = iteration

            best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]
            print("Final distance " + str(dist))
            print("Final relative distance " + str(dist/original_distance))
            print("Min relative distance " + str(min_dist/original_distance))
            print("Min iter: " + str(min_iter))
            print("Final iter: " + str(iteration))
            print("Min distance " + str(min_dist))

            self.f_iterations = iteration
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

            # self.plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
