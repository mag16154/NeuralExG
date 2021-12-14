from tensorflow.keras.models import load_model
import numpy as np
from learningModule import DataConfiguration
from os import path
from mpl_toolkits import mplot3d
from rbflayer import RBFLayer, InitCentersRandom


class Evaluation(object):
    def __init__(self, dynamics, sensitivity, dnn_rbf, layers, neurons, act_fn, grad_run):
        self.data_object = None
        self.debug_print = False
        self.dynamics = dynamics
        self.iter_count = 10
        self.eval_dir = '../../../eval-hscc/eval-gr'
        if grad_run is False:
            self.eval_dir = '../../../eval-hscc/eval-non-gr'
        self.sensitivity = sensitivity
        self.network = self.EvalNetwork(dnn_rbf, layers, neurons, act_fn)

    def setEvalDir(self, evalDir):
        self.eval_dir = evalDir

    def getDataObject(self):
        assert self.data_object is not None
        return self.data_object

    def setDataObject(self, d_obj_f_name=None):

        if d_obj_f_name is None and self.sensitivity is 'Inv':
            d_obj_f_name = self.eval_dir + '/dconfigs_inv/d_object_'+self.dynamics
        elif d_obj_f_name is None and self.sensitivity is 'Fwd':
            d_obj_f_name = self.eval_dir + '/dconfigs_fwd/d_object_'+self.dynamics

        d_obj_f_name = d_obj_f_name + '.txt'
        print(d_obj_f_name)

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

    def setIterCount(self, iter_count):
        self.iter_count = iter_count

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

    def getModel(self):
        trained_model = self.network.getNetworkModel(self.eval_dir, self.sensitivity, self.dynamics)
        return trained_model

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

    def dumpModel(self):
        model_file = self.eval_dir + '/models/controllerTora_nnv.h5'
        model_v = load_model(model_file, compile=False)
        yaml_model = model_v.to_yaml()
        with open('controllerTora.yaml', 'w') as yaml_file:
            yaml_file.write(yaml_model)

    class EvalNetwork:
        def __init__(self, dnn_rbf, layers, neurons, act_fn):
            self.dnn_rbf = dnn_rbf
            self.layers = layers
            self.neurons = neurons
            self.act_fn = act_fn

        def getNetworkModel(self, eval_dir, sensitivity, dynamics):
            model_f_name = eval_dir
            if sensitivity is 'Fwd':
                model_f_name = model_f_name + '/models/model_v_2_vp_'
            else:
                model_f_name = model_f_name + '/models/convergence-more-data/model_vp_2_v_'
            model_f_name = model_f_name + dynamics + "_" + self.dnn_rbf
            model_f_name = model_f_name + "_" + str(self.layers)
            model_f_name = model_f_name + "_" + str(self.neurons)
            model_f_name = model_f_name + "_" + self.act_fn
            model_f_name = model_f_name + '.h5'
            trained_model = None
            print(model_f_name)
            if path.exists(model_f_name):
                if self.dnn_rbf is 'dnn':
                    trained_model = load_model(model_f_name, compile=True)
                else:
                    trained_model = load_model(model_f_name, compile=True, custom_objects={'RBFLayer': RBFLayer})
            else:
                print("Model file " + model_f_name + " does not exists.")

            return trained_model



