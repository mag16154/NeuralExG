from evaluation import Evaluation
from sampler import generateRandomStates
from matplotlib.path import Path
import matplotlib.patches as patches
import time
import numpy as np
import matplotlib.pyplot as plt
from frechet import norm
from evaluation_plot import plotInvSenResults, plotInvSenResultsAnimate
import random

# version 1 RRT
from rrtv1 import RRTV1

# version 2 RRT
import sys
sys.path.append('./rrtAlgorithms/src/')
from rrt.rrt import RRT
from search_space.search_space import SearchSpace
from utilities.plotting import Plot


class EvaluationInvSen(Evaluation):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU'):
        Evaluation.__init__(self, dynamics=dynamics, sensitivity='Inv', dnn_rbf=dnn_rbf, layers=layers,
                            neurons=neurons, act_fn=act_fn, grad_run=True)
        self.f_iterations = 0
        self.f_dist = None
        self.f_rel_dist = None
        self.staliro_run = False
        self.usafelowerBoundArray = []
        self.usafeupperBoundArray = []
        self.usafe_centroid = None
        self.time_steps = []
        self.always_spec = False

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
        self.usafe_centroid = np.zeros(self.data_object.dimensions)
        for dim in range(self.data_object.dimensions):
            self.usafe_centroid[dim] = (lowerBound[dim] + upperBound[dim])/2
        self.staliro_run = True

    def check_for_usafe_contain_always(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        is_contained = True
        for time_step in time_steps:
            state = traj[time_step]
            for dim in range(self.data_object.dimensions):
                l_bound = self.usafelowerBoundArray[dim]
                u_bound = self.usafeupperBoundArray[dim]
                if l_bound < state[dim] < u_bound:
                    continue
                else:
                    is_contained = False
                    break

            if is_contained is False:
                break

        if is_contained:
            found_time_step = time_steps[1]
            print(" Trajectory found ")

        return found_time_step

    def check_for_usafe_contain_eventual(self, traj):
        # print(" Checking for containment " + str(state))
        time_steps = self.time_steps
        # time_steps = [d_time_step]
        found_time_step = -1
        for time_step in time_steps:
            is_contained = True
            state = traj[time_step]
            for dim in range(self.data_object.dimensions):
                l_bound = self.usafelowerBoundArray[dim]
                u_bound = self.usafeupperBoundArray[dim]
                if l_bound < state[dim] < u_bound:
                    continue
                else:
                    is_contained = False
                    break

            if is_contained:
                found_time_step = time_step
                # print(" Found time step " + str(found_time_step))
                break

        return found_time_step

    def compute_robust_state_wrt_axes(self, state):

        robustness = 100.0

        for dim in range(self.data_object.dimensions):
            l_bound = self.usafeupperBoundArray[dim]
            u_bound = self.usafeupperBoundArray[dim]

            dist_1 = abs(state[dim] - l_bound)
            dist_2 = abs(u_bound - state[dim])

            if dist_1 < robustness:
                robustness = dist_1
            if dist_2 < robustness:
                robustness = dist_2

        return robustness

    def compute_robust_wrt_axes(self, traj):

        found_time_step = self.check_for_usafe_contain_eventual(traj)
        if self.always_spec is True:
            found_time_step = self.check_for_usafe_contain_always(traj)

        robustness = 100

        if found_time_step != -1:
            state = traj[found_time_step]
            robustness = self.compute_robust_state_wrt_axes(state)
        else:
            # state = traj[self.time_steps[int(len(self.time_steps)/2)]]
            for time_step in self.time_steps:
                state = traj[time_step]
                cur_robustness = self.compute_robust_state_wrt_axes(state)
                if cur_robustness < robustness:
                    robustness = cur_robustness
        # robustness = 100.0
        #
        # for dim in range(self.data_object.dimensions):
        #     l_bound = self.usafeupperBoundArray[dim]
        #     u_bound = self.usafeupperBoundArray[dim]
        #
        #     dist_1 = abs(state[dim] - l_bound)
        #     dist_2 = abs(u_bound - state[dim])
        #
        #     if found_time_step != -1 or state[dim] < l_bound or state[dim] > u_bound:
        #         if dist_1 < robustness:
        #             robustness = dist_1
        #         if dist_2 < robustness:
        #             robustness = dist_2

        if found_time_step != -1:
            robustness = robustness * -1

        return robustness

    def generateRandomUnsafeStates(self, samples):
        states = generateRandomStates(samples, self.usafelowerBoundArray, self.usafeupperBoundArray)
        return states

    # For rrt algorithms repo (RRT v2) based path. Plot them in html
    def plotInvSenResultsRRTv2(self, trajectories, destination, d_time_step, dimensions, rand_area, path_idx, dynamics):

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
                          scaling_factors=[0.01], i_state=None, adaptation_run=False, rand_area=None, n_paths=2):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]
        d_time_step = d_time_steps[0]

        assert self.data_object is not None

        dimensions = self.data_object.getDimensions()

        trained_model = self.getModel()

        if trained_model is None:
            return

        paths_list = []

        for p_idx in range(n_paths):
            rrt_path_points = []
            # obstacleList = [
            #     (1.75, -0.25, 0.1)
            # ]  # [x, y, radius]
            obstacleList = []
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
                Q = np.array([(10, 2)])  # length of tree edges
                r = 20  # length of smallest edge to check for intersection with obstacles
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
                print(len(rrt_path_points))

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
                                                  threshold=threshold, model_v=trained_model, iter_bound=self.iter_count,
                                                  scaling_factor=s_factor, adaptation_run=adaptation_run,
                                                  adaptation_factor=adaptation_factor, dynamics=self.dynamics)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvNonBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold, model_v=trained_model, correction_steps=steps,
                                                     scaling_factor=s_factor, iter_bound=self.iter_count,
                                                     adaptation_run=adaptation_run, adaptation_factor=adaptation_factor)
                        print("Time taken: " + str(time.time() - start_time))

    def predict_falsifying_time_step(self, dest, traj, mid=True):
        d_time_step = -1
        if mid is True:
            d_time_step = int((self.time_steps[0] + self.time_steps[len(self.time_steps)-1])/2)
            # d_time_step = random.randint(d_time_steps[0], d_time_steps[1])
        else:
            min_dist = 100
            for t_idx in range(self.time_steps[0], self.time_steps[len(self.time_steps)-1]):
                current_dist = norm(dest-traj[t_idx], 2)
                if current_dist < min_dist:
                    min_dist = current_dist
                    d_time_step = t_idx
        return d_time_step

    def reachDestInvPaths(self, dests=None, d_time_steps=None, threshold=0.01, correction_steps=[50],
                          scaling_factors=[0.01], i_state=None, adaptation_run=False):

        ref_traj = self.data_object.generateTrajectories(r_states=i_state)[0]
        dest = dests[0]

        assert self.data_object is not None

        trained_model = self.getModel()
        if trained_model is None:
            return

        # original_vector = dest - ref_traj[d_time_step]
        # original_vector_div = original_vector/3
        # path1_src = ref_traj[d_time_step]
        # path1_dest = ref_traj[d_time_step] + original_vector_div
        # path1 = [path1_src, path1_dest]
        # path2_dest = path1_dest + original_vector_div
        # path2 = [path1_dest, path2_dest]
        # path3 = [path2_dest, dest]
        # # paths_list = [[path1, path2, path3]]

        if d_time_steps is None:
            print(" Provide a valid time step ")
            return
        elif len(d_time_steps) == 1:
            d_time_step = d_time_steps[0]
        else:
            for t_idx in range(d_time_steps[0]+1, d_time_steps[1]+1):
                self.time_steps.append(t_idx)
            d_time_step = self.predict_falsifying_time_step(dest, ref_traj, False)
        print(d_time_step)
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

        # f_iterations = None
        # min_dist = None
        # rel_dist = None

        for paths in paths_list:
            # print(paths)
            for s_factor in scaling_factors:

                adaptation_factor = 1 / 2
                if s_factor < 0.01:
                    adaptation_factor = 3/4

                for steps in correction_steps:
                    if steps == 1:
                        # print(" *** Baseline Greedy *** \n")
                        # start_time = time.time()
                        # self.reachDestInvBaselineGreedy(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                        #                                 threshold=threshold, model_v=trained_model,
                        #                                 iter_bound=self.iter_count,
                        #                                 scaling_factor=s_factor, adaptation_run=adaptation_run,
                        #                                 adaptation_factor=adaptation_factor, dynamics=self.dynamics)
                        # print("Time taken: " + str(time.time() - start_time))
                        start_time = time.time()
                        self.reachDestInvBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                  threshold=threshold, model_v=trained_model, iter_bound=self.iter_count,
                                                  scaling_factor=s_factor, adaptation_run=adaptation_run,
                                                  adaptation_factor=adaptation_factor, dynamics=self.dynamics)
                        print("Time taken: " + str(time.time() - start_time))
                    else:
                        start_time = time.time()
                        self.reachDestInvNonBaseline(ref_traj=ref_traj, paths=paths, d_time_step=d_time_step,
                                                     threshold=threshold,  model_v=trained_model, correction_steps=steps,
                                                     scaling_factor=s_factor, iter_bound=self.iter_count,
                                                     adaptation_run=adaptation_run, adaptation_factor=adaptation_factor,
                                                     dynamics=self.dynamics)
                        print("Time taken: " + str(time.time() - start_time))

        # if self.staliro_run is True:
        #     return f_iterations, min_dist, rel_dist

    '''
    ReachDestination for correction period 1 (without greedy).
    It can be removed later as reachDestInvNonBaseline works just fine for correction period = 1 as well.
    '''
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
            before_adaptation_iter = 1
            after_adaptation_iter = 0
            adaptation_applied = False
            prev_dist_while_adaptation = None
            v_val = dest - x_val
            print(xp_val, dest)
            vp_val = dest - xp_val
            vp_norm = norm(vp_val, 2)
            min_iter = 0
            t_val = d_time_step
            vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
            vp_val_scaled = [val * scaling_factor for val in vp_val_normalized]
            print(iter_bound)
            dist = vp_norm
            print("Starting distance: " + str(dist))
            final_dist = dist
            original_distance = final_dist
            min_dist = final_dist
            best_trajectory = ref_traj
            adaptations = 0
            halt_adaptation = False

            # These 2 lines were used to test the routine without course correction
            # x_vals = [x_val]
            # xp_vals = [xp_val]
            vp_vals = [vp_val_scaled]
            v_vals = [v_val]
            while final_dist > threshold and (before_adaptation_iter + after_adaptation_iter) < iter_bound:
                data_point = self.data_object.createDataPoint(x_val, xp_val, v_val, vp_val_normalized, t_val)
                predicted_v = self.evalModel(input=data_point, eval_var='v', model=model_v)
                predicted_v_scaled = [val * scaling_factor for val in predicted_v]
                new_init_state = [self.check_for_bounds(x_val + predicted_v_scaled)]
                new_traj = self.data_object.generateTrajectories(r_states=new_init_state)[0]
                x_val = new_traj[0]
                xp_val = new_traj[d_time_step]

                # These 4 lines were used to test the routine without course correction
                # x_val = new_init_state[0]
                # xp_val = xp_val + vp_val_scaled
                # x_vals.append(x_val)
                # xp_vals.append(xp_val)
                v_val = predicted_v_scaled
                vp_val = dest - xp_val
                vp_norm = norm(vp_val, 2)
                dist = vp_norm
                vp_val_normalized = [val / vp_norm for val in vp_val]  # Normalized
                vp_val_temp = [val * scaling_factor for val in vp_val_normalized]
                vp_vals.append(vp_val_temp)
                v_vals.append(predicted_v_scaled)
                t_val = d_time_step

                if adaptation_applied is False:
                    before_adaptation_iter = before_adaptation_iter + 1
                else:
                    after_adaptation_iter = after_adaptation_iter + 1

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    best_trajectory = new_traj
                    if adaptation_applied is False:
                        min_iter = before_adaptation_iter
                    else:
                        min_iter = before_adaptation_iter + after_adaptation_iter

                if adaptation_run is True and final_dist is not None and (final_dist - dist > 1e-7) \
                        and halt_adaptation is False:
                    # print("Changing the scaling factor")
                    if original_scaling_factor >= 0.01:
                        scaling_factor += 0.0002
                    else:
                        scaling_factor += 0.0001

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
            # best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]
            print("Final distance " + str(final_dist))
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

            plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # plotInvSenResultsAnimate(trajectories, rrt_dests, d_time_step, v_vals, vp_vals)
            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx,dynamics)

    '''
    Reach Destination implementation (without greedy).
    '''
    def reachDestInvNonBaseline(self, ref_traj, paths, d_time_step, threshold, model_v, correction_steps, iter_bound,
                                scaling_factor, adaptation_factor, adaptation_run, rand_area=None, dynamics=None):

        original_scaling_factor = scaling_factor
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
            before_adaptation_iter = 1
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
            # best_state = x_val
            best_trajectory = ref_traj
            halt_adaptation = False
            min_iter = 0

            while final_dist > threshold and (before_adaptation_iter + after_adaptation_iter) < iter_bound:

                if self.staliro_run is True and before_adaptation_iter < 2:

                    if self.always_spec is False and self.check_for_usafe_contain_eventual(ref_traj) != -1:
                        print("*********** Initial sample falsified ************")
                        break
                    elif self.always_spec is True and self.check_for_usafe_contain_always(ref_traj) != -1:
                        print("*********** Initial sample falsified ************")
                        break

                # if self.staliro_run is True and 20 == (before_adaptation_iter + after_adaptation_iter):
                #     # How about pick up a different dest in U?
                #     d_time_step = self.predict_falsifying_time_step(dest, best_trajectory, True)
                #     # if d_time_step < int((self.time_steps[0] + self.time_steps[len(self.time_steps)-1])/2):
                #     #     d_time_step = d_time_step + 2
                #     # else:
                #     #     d_time_step = d_time_step - 2
                #     print("Changed the time step to " + str(d_time_step))
                #     xp_val = best_trajectory[d_time_step]
                #     vp_val = dest - xp_val
                #     vp_norm = norm(vp_val, 2)

                if adaptation_applied is False:
                    before_adaptation_iter = before_adaptation_iter + 1
                else:
                    after_adaptation_iter = after_adaptation_iter + 1

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

                trajectories.append(new_traj)

                if dist < min_dist:
                    min_dist = dist
                    # best_state = x_val
                    best_trajectory = new_traj
                    if adaptation_applied is False:
                        min_iter = before_adaptation_iter
                    else:
                        min_iter = before_adaptation_iter + after_adaptation_iter

                if adaptation_run is True and final_dist is not None and (final_dist - dist > 1e-6) and \
                        halt_adaptation is False:
                    # print("Changing the scaling factor because distance is " + str(final_dist) + str(dist))
                    if original_scaling_factor >= 0.01:
                        scaling_factor += 0.0002
                    else:
                        scaling_factor += 0.0001

                elif adaptation_run is True and dist >= final_dist and halt_adaptation is False:
                    current_rel_dist = final_dist / original_distance
                    if adaptations == 2 or (prev_dist_while_adaptation is not None and
                                            current_rel_dist > prev_dist_while_adaptation):
                        # trajectories.pop()

                        if self.staliro_run is False:
                            print("Adaptations reached it's limits")
                        halt_adaptation = True
                        # if self.staliro_run is True:
                        #     break
                    else:
                        # print("Before adaptation relative distance " + str(current_rel_dist))
                        if prev_dist_while_adaptation is None:
                            prev_dist_while_adaptation = current_rel_dist
                        adaptation_applied = True
                        scaling_factor = scaling_factor * adaptation_factor
                        adaptations += 1

                final_dist = dist

                # A falsifying trajectory found
                if self.staliro_run and self.always_spec is False and self.check_for_usafe_contain_eventual(new_traj) != -1:
                    best_trajectory = new_traj
                    # print("Found the time step **** ")
                    break
                elif self.staliro_run and self.always_spec is True and self.check_for_usafe_contain_always(new_traj) != -1:
                    best_trajectory = new_traj
                    print("Found the time step **** ")
                    break
                elif self.staliro_run and self.always_spec is True and self.check_for_usafe_contain_always(new_traj) == -1:
                    new_dest = self.generateRandomUnsafeStates(1)[0]
                    new_time_step = self.predict_falsifying_time_step(new_dest, new_traj, False)
                    new_dist = norm(new_traj[new_time_step]-new_dest, 2)
                    if new_dist < min_dist:
                        print("Setting new time step to " + str(new_time_step))
                        d_time_step = new_time_step
                        best_trajectory = new_traj
                        min_dist = new_dist
                        dest = new_dest

            min_rel_dist = min_dist / original_distance
            # best_trajectory = self.data_object.generateTrajectories(r_states=[best_state])[0]

            if self.staliro_run:
                min_dist = self.compute_robust_wrt_axes(best_trajectory)
                print("Best robustness " + str(min_dist))

            if self.staliro_run is False:
                print("Final distance " + str(final_dist))
                print("Final relative distance " + str(final_dist / original_distance))
                print("Min relative distance: " + str(min_rel_dist))
                print("Min iter: " + str(min_iter))
            print("Before adaptation iterations: " + str(before_adaptation_iter))
            print("Final iterations: " + str(before_adaptation_iter + after_adaptation_iter))
            self.f_iterations = before_adaptation_iter + after_adaptation_iter
            self.f_dist = min_dist
            self.f_rel_dist = min_rel_dist

            # self.plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx, dynamics)

            # if self.staliro_run:
            #     self.plotInvSenStaliroResults(trajectories, d_time_step, best_trajectory)

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

    '''
    ReachDestination with Greedy for correction period 1
    '''
    def reachDestInvBaselineGreedy(self, ref_traj, paths, d_time_step, threshold, model_v, iter_bound, scaling_factor,
                                   adaptation_factor, adaptation_run, dynamics=None):

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
                    if original_scaling_factor >= 0.01:
                        scaling_factor += 0.0002
                    else:
                        scaling_factor += 0.0001

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
            print("Min dist: " + str(min_dist))
            print("Min iter: " + str(min_iter))
            # print("Terminating state: " + str(x_val))
            print("Before adaptation iterations: " + str(before_adaptation_iter))
            print("Final iterations: " + str(before_adaptation_iter + after_adaptation_iter))
            self.f_iterations = before_adaptation_iter + after_adaptation_iter
            self.f_dist = min_dist
            self.f_rel_dist = min_dist/original_distance

            # plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResults(trajectories, rrt_dests, d_time_step, dimensions, best_trajectory)
            # self.plotInvSenResultsRRTv2(trajectories, dest, d_time_step, dimensions, rand_area, path_idx,dynamics)
