from evaluation import Evaluation
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from circleRandom import generate_points_in_circle
import random
from frechet import norm
import numpy as np
import math


class EvaluationFwdSen(Evaluation):

    def __init__(self, dynamics='None', layers=1, neurons=512, dnn_rbf='RBF', act_fn='ReLU'):
        Evaluation.__init__(self, dynamics=dynamics, sensitivity='Fwd', layers=layers, neurons=neurons, dnn_rbf=dnn_rbf,
                            act_fn=act_fn)

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

    def plotFwdSenTrajectoriesNew(self, ref_trajs, predicted_trajs, actual_trajs, max_time, x_vals=None, xp_vals=None):
        x_index = 0
        y_index = 1

        i_x_min = self.data_object.lowerBoundArray[x_index]
        i_x_max = self.data_object.upperBoundArray[x_index]
        i_y_min = self.data_object.lowerBoundArray[y_index]
        i_y_max = self.data_object.upperBoundArray[y_index]

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

        fig, ax = plt.subplots()

        i_path = Path(i_verts, codes)

        i_patch = patches.PathPatch(i_path, facecolor='red', lw=1, fill=False)
        ax.add_patch(i_patch)

        n_trajs = len(predicted_trajs)

        # ax.plot(ref_trajs[0][0:max_time, x_index], ref_trajs[0][0:max_time, y_index], color='b',
        #         label='Reference trajectory')
        ax.plot(predicted_trajs[n_trajs-1][0:max_time, x_index], predicted_trajs[n_trajs-1][0:max_time, y_index],
                color='r', label='Predicted trajectory')
        ax.plot(actual_trajs[n_trajs-1][0:max_time, x_index], actual_trajs[n_trajs-1][0:max_time, y_index], color='g',
                label='Actual trajectory')
        for idx in range(0, n_trajs):
            # ref_traj = ref_trajs[idx]
            # ax.plot(ref_traj[0:max_time, x_index], ref_traj[0:max_time, y_index], color='b')
            predicted_traj = predicted_trajs[idx]
            actual_traj = actual_trajs[idx]
            ax.plot(predicted_traj[0:max_time, x_index], predicted_traj[0:max_time, y_index], color='r')
            ax.plot(actual_traj[0:max_time, x_index], actual_traj[0:max_time, y_index], color='g')

        # if x_vals is not None and len(x_vals) > 0:
        #     ax.scatter(x_vals[0][0], x_vals[0][1], color='g', label='Initial states')
            # ax.scatter(xp_vals[0][0], xp_vals[0][1], color='r', label='Pred destination states')
            # for idx in range(1, len(x_vals)):
            #       ax.scatter(x_vals[idx][0], x_vals[idx][1], color='g')
            #       ax.scatter(xp_vals[idx][0], xp_vals[idx][1], color='r')

        ax.set_xlabel('x' + str(x_index))
        ax.set_ylabel('x' + str(y_index))
        plt.legend()
        plt.show()

    '''Generate as many reference trajectories as number of delta vectors. Each delta vector is 
    applied to a different reference trajectory.'''

    def spaceExploreFwdMultipleRef(self, scaling_factor=0.01, d_time_step=10):

        n_vectors = 1
        max_time = 200
        assert self.data_object is not None

        trained_model = self.getModel()
        if trained_model is None:
            return

        vecs_in_unit_circle = generate_points_in_circle(n_samples=n_vectors, dim=self.data_object.dimensions)

        delta_vecs = []
        for vec in vecs_in_unit_circle:
            delta_vec = [val * scaling_factor for val in vec]
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
            for idy in range(0, 300):
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
                    predicted_vp = self.evalModel(input=data_point, eval_var='vp', model=trained_model)
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
        self.plotFwdSenTrajectoriesNew(original_ref_trajs, pred_trajs, actual_trajs, max_time, x_vals, xp_vals)

    '''Generate n_vectors number of delta vectors. Start from just one reference trajectory. The next reference 
    trajectory is the one predicted after applying first delta vectors. The final trajectory is plotted
    after applying all vectors to the initial reference trajectory.'''

    def spaceExploreFwdOneRef(self, scaling_factor=0.01):

        n_vectors = 3
        max_time = 250
        assert self.data_object is not None

        trained_model = self.getModel()
        if trained_model is None:
            return

        vecs_in_unit_circle = generate_points_in_circle(n_samples=10, dim=self.data_object.dimensions)

        delta_vecs = []
        for vec in vecs_in_unit_circle:
            delta_vec = [val * scaling_factor for val in vec]
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
            for idy in range(0, 100):
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
                    predicted_vp = self.evalModel(input=data_point, eval_var='vp', model=trained_model)
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
            self.plotFwdSenTrajectoriesNew(original_ref_trajs, pred_trajs, actual_trajs, max_time, x_vals, xp_vals)

    def spaceExploreFwdReachability(self, scaling_factor=0.001, destinations=None, ref_state=None):

        if destinations is None:
            destinations = []

        x_index = 0
        y_index = 1
        i_x_min = self.data_object.lowerBoundArray[x_index]
        i_x_max = self.data_object.upperBoundArray[x_index]
        i_y_min = self.data_object.lowerBoundArray[y_index]
        i_y_max = self.data_object.upperBoundArray[y_index]
        i_x_mid = (i_x_min + i_x_max) / 2
        i_y_mid = (i_y_min + i_y_max) / 2

        dimesnsions = self.data_object.getDimensions()

        i_z_min = i_x_min
        i_z_max = i_x_max
        i_w_min = i_x_min
        i_w_max = i_x_max
        i_z_mid = i_x_mid
        i_w_mid = i_x_mid
        if dimesnsions == 3:
            z_index = 2
            i_z_min = self.data_object.lowerBoundArray[z_index]
            i_z_max = self.data_object.upperBoundArray[z_index]
            i_z_mid = (i_z_min + i_z_max) / 2
        elif dimesnsions == 4:
            w_index = 3
            i_w_min = self.data_object.lowerBoundArray[w_index]
            i_w_max = self.data_object.upperBoundArray[w_index]
            i_w_mid = (i_w_min + i_w_max) / 2

        if ref_state is None:
            if dimesnsions == 2:
                ref_state = [i_x_mid, i_y_mid]
            elif dimesnsions == 3:
                ref_state = [i_x_mid, i_y_mid, i_z_mid]
            elif dimesnsions == 4:
                ref_state = [i_x_mid, i_y_mid, i_z_mid, i_w_mid]

        if dimesnsions == 2:
            destinations.append([i_x_min, i_y_min])
            destinations.append([i_x_max, i_y_min])
            destinations.append([i_x_max, i_y_max])
            destinations.append([i_x_min, i_y_max])
        elif dimesnsions == 3:
            destinations.append([i_x_min, i_y_min, i_z_min])
            destinations.append([i_x_max, i_y_min, i_z_min])
            destinations.append([i_x_max, i_y_max, i_z_max])
            destinations.append([i_x_min, i_y_max, i_z_max])
        elif dimesnsions == 4:
            # destinations.append([i_x_min, i_y_min, i_z_min, i_w_min])
            # destinations.append([i_x_max, i_y_min, i_z_min, i_w_min])
            # destinations.append([i_x_max, i_y_max, i_z_max, i_w_max])
            destinations.append([i_x_min, i_y_max, i_z_max, i_w_max])

        n_vectors = len(destinations)

        max_time = self.data_object.steps
        assert self.data_object is not None

        trained_model = self.getModel()

        if trained_model is None:
            return

        # vecs_in_unit_circle = generate_points_in_circle(n_samples=10, dim=self.data_object.dimensions)

        pred_trajs = []
        actual_trajs = []
        original_ref_trajs = []
        ref_traj = self.data_object.generateTrajectories(r_states=[ref_state])[0]
        original_ref_trajs.append(ref_traj)
        plt.plot(ref_traj[0:max_time, 0], ref_traj[0:max_time, 1], 'b')
        plt.show()

        xp_vals = []
        x_vals = []
        d_time_step = random.randint(20, 100)
        for idx in range(n_vectors):
            ref_traj = original_ref_trajs[0]
            x_val = ref_traj[0]
            x_vals.append(x_val)
            xp_vals.append(ref_traj[d_time_step])

            destination = destinations[idx]
            v_val = [destination[i] - ref_state[i] for i in range(len(ref_state))]
            v_val_norm = norm(v_val, 2)
            v_val_unit_vec = [val / v_val_norm for val in v_val]
            iterations = math.ceil(v_val_norm / scaling_factor)
            print(iterations)

            delta_vec = [val * scaling_factor for val in v_val_unit_vec]
            delta_vec_norm = norm(delta_vec, 2)
            # delta_vec_scaled = [val / delta_vec_norm for val in delta_vec]
            # print(v_val_scaled, delta_vec_unit_vec)
            for idy in range(0, iterations):
                neighbor_state = [x_val[i] + delta_vec[i] for i in range(len(delta_vec))]
                pred_traj = [neighbor_state]
                for t_step in range(1, max_time):
                    # print(t_step)
                    xp_val = ref_traj[t_step]
                    t_val = t_step
                    vp_val = v_val
                    data_point = self.data_object.createDataPoint(x_val, xp_val, v_val_unit_vec, vp_val, t_val)
                    predicted_vp = self.evalModel(input=data_point, eval_var='vp', model=trained_model)
                    predicted_vp = predicted_vp * delta_vec_norm
                    pred_dest = [xp_val[i] + predicted_vp[i] for i in range(len(xp_val))]
                    pred_traj.append(pred_dest)
                ref_traj = pred_traj
                x_val = neighbor_state
                x_vals.append(x_val)
                xp_vals.append(ref_traj[d_time_step])

            print(x_vals[len(x_vals)-1], destination, ref_traj[0])
            print("Done**** " + str(idx))
            ref_traj = np.array(ref_traj)
            pred_trajs.append(ref_traj)
            actual_traj = self.data_object.generateTrajectories(r_states=[ref_traj[0]])[0]
            actual_trajs.append(actual_traj)
        self.plotFwdSenTrajectoriesNew(original_ref_trajs, pred_trajs, actual_trajs, max_time, x_vals, xp_vals)