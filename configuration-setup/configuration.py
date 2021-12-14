import sys
import numpy as np
# sys.path.append('../AeroBenchVVPython/')
# sys.path.append('../AeroBenchVVPython/code')
# from aeroBenchSys import AeroBenchSim

from sampler import generateRandomStates, generateSuperpositionSampler
from ODESolver import generateTrajectories, plotTrajectories
from nnControllerHybridSystems import DnnController, Plant
from nonNNController import nonNNPlant
import random
from os import path
import os.path
from circleRandom import generate_points_in_circle


class configuration(object):
    def __init__(self, timeStep=0.01, steps=100, samples=50, dynamics='None', dimensions=2, lowerBound=[],
                upperBound=[], gradient_run=False, embedding_dimension=2):
        self.timeStep = timeStep
        self.steps = steps
        self.samples = samples
        self.dynamics = dynamics
        self.dimensions = dimensions
        self.lowerBoundArray = lowerBound
        self.upperBoundArray = upperBound
        self.trajectories = []
        self.states = []
        self.neighbors = 10
        self.grad_run = gradient_run

    def setTrajectories(self, trajectories):
        self.trajectories = trajectories

    def getTrajectories(self):
        return self.trajectories

    def setDynamics(self, dynamics):
        self.dynamics = dynamics

    def setLowerBound(self, lowerBound):
        self.lowerBoundArray = lowerBound

    def setUpperBound(self, upperBound):
        self.upperBoundArray = upperBound

    def setSamples(self, samples):
        self.samples = samples

    def setSteps(self, steps):
        self.steps = steps

    def setTimeStep(self, timeStep):
        self.timeStep = timeStep

    def setEmbeddingDimension(self, dim):
        self.embedding_dimension = dim

    def getDynamics(self):
        return self.dynamics

    # Only for gradient run
    def setNeighbors(self, neighbors):
        assert self.grad_run is True
        self.neighbors = neighbors

    def getGradientRun(self):
        return self.grad_run

    def setGradientRun(self):
        self.grad_run = True

    def generateTrajectories4GradRun(self, scaling, samples, r_states):
        plant = None
        yml_file_dir = '../../../controllers-yml-files/'
        if self.dynamics is 'MC':
            dnn_cntrl_fname = yml_file_dir + 'verisig/mountain_car/' + 'sig16x16.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
        elif self.dynamics is 'ABS':
            dnn_cntrl_fname = yml_file_dir + 'NNV/ABS/' + 'controller.yml'
            dnn_tf_fname = yml_file_dir + 'NNV/ABS/' + 'transform.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            dnn_transform_obj = DnnController(dnn_tf_fname, 2)
            plant = Plant(self.dynamics, dnn_controller_obj, dnn_transform_obj, self.steps)
        elif self.dynamics is 'Quadrotor':
            dnn_cntrl_fname = yml_file_dir + 'verisig/quadrotor/' + 'tanh20x20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
        elif self.dynamics is 'InvPendulumD':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Inverted_pendulum/' + 'IPController.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('IPendulumD', dnn_controller_obj, None, self.steps)
        elif self.dynamics is 'OtherBenchDisc1':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark1/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('OBenchmarkDisc1', dnn_controller_obj, None, self.steps)
        elif self.dynamics is 'OtherBenchC1':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark1/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC2':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark2/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC3':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark3/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC4':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark4/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC5':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark5/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC6':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019//Benchmark6/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC7':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark7/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC8':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Benchmark8/' + 'controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'ACCNonLinear3L':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/ACC/' + 'acc_controller_3_20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'ACCNonLinear5L':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/ACC/' + 'acc_controller_5_20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'ACCNonLinear7L':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/ACC/' + 'acc_controller_7_20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'ACCNonLinear10L':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/ACC/' + 'acc_controller_10_20.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC9':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/Benchmark9-TORA/' + 'controllerTora_nnv.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC9Tanh':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/Benchmark9-TORA-Tanh/' + 'controllerTora_nnv_tanh.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC9Sigmoid':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/Benchmark9-TORA-Sigmoid/' + 'controllerTora_nnv_sigmoid.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'SinglePendulum':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/SinglePendulum/' + 'controller_single_pendulum.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'DoublePendulumLess':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/DoublePendulum/' + 'controller_double_pendulum_less.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'DoublePendulumMore':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/DoublePendulum/' + 'controller_double_pendulum_more.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'OtherBenchC10':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/Benchmark10-Unicycle/' + 'controller10_unicycle.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'InvPendulumC':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Inverted_pendulum/' + 'IPController.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'Airplane':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/Airplane/' + 'controller_airplane.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'CartPole':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Cart-pole/' + 'Cartpole_controller.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'CartPoleTanh':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2019/Cart-pole/' + 'Cartpole_controller_tanh.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant(self.dynamics, dnn_controller_obj, None, self.steps)
            plant.dnn_controllers[0].parseDNNYML(self.dynamics)
        elif self.dynamics is 'CartPoleLinControl':
            plant = nonNNPlant('CartPoleLinControl')
        elif self.dynamics is 'CartPoleLinControl2':
            plant = nonNNPlant('CartPoleLinControl2')
        elif self.dynamics is 'BouncingBall':
            plant = nonNNPlant('BouncingBall')
        elif self.dynamics is 'VertCAS':
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_1.yml'
            dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
            plant = Plant('VertCAS', dnn_controller_obj, None, self.steps)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_2.yml'
            dnn_controller_obj2 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj2)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_3.yml'
            dnn_controller_obj3 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj3)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_4.yml'
            dnn_controller_obj4 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj4)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_5.yml'
            dnn_controller_obj5 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj5)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_6.yml'
            dnn_controller_obj6 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj6)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_7.yml'
            dnn_controller_obj7 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj7)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_8.yml'
            dnn_controller_obj8 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj8)
            dnn_cntrl_fname = yml_file_dir + 'ARCH-2020/VCAS/' + 'VertCAS_9.yml'
            dnn_controller_obj9 = DnnController(dnn_cntrl_fname, self.dimensions)
            plant.append_controller(dnn_controller_obj9)

        if r_states is not None:
            if plant is not None and (self.dynamics is 'MC' or self.dynamics is 'ABS' or self.dynamics is 'Quadrotor'
                                      or self.dynamics is 'OtherBenchDisc1' or self.dynamics is 'InvPendulumD'
                                      or self.dynamics is 'VertCAS'):
                r_trajectories = plant.getSimulations(states=r_states)
            else:
                r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
                r_trajectories = generateTrajectories(self.dynamics, r_states, r_time, plant)
            return r_trajectories
        else:
            if samples is not None:
                r_states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                if plant is not None and (self.dynamics is 'MC' or self.dynamics is 'ABS' or
                                          self.dynamics is 'Quadrotor' or self.dynamics is 'OtherBenchDisc1'
                                          or self.dynamics is 'InvPendulumD' or self.dynamics is 'VertCAS'):
                    r_trajectories = plant.getSimulations(states=r_states)
                else:
                    # r_states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
                    r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
                    r_trajectories = generateTrajectories(self.dynamics, r_states, r_time, plant)
                self.trajectories = r_trajectories
                return self.trajectories
            else:
                r_states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
                if plant is not None and (self.dynamics is 'MC' or self.dynamics is 'ABS'
                                          or self.dynamics is 'Quadrotor' or self.dynamics is 'OtherBenchDisc1'
                                          or self.dynamics is 'InvPendulumD' or self.dynamics is 'VertCAS'):
                    r_trajectories = plant.getSimulations(states=r_states)
                else:
                    r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
                    r_trajectories = generateTrajectories(self.dynamics, r_states, r_time, plant)

                if self.grad_run is True:
                    vecs_in_unit_circle = generate_points_in_circle(n_samples=self.neighbors, dim=self.dimensions)
                    # print(vecs_in_unit_circle)
                    delta_vecs = []
                    for vec in vecs_in_unit_circle:
                        delta_vec = [val*scaling for val in vec]
                        delta_vecs.append(delta_vec)
                    # print(delta_vecs)
                    trajectories = []
                    for idx in range(len(r_states)):
                        trajectories.append(r_trajectories[idx])
                        # print(r_trajectories[idx])
                        r_state = r_states[idx]
                        for delta_vec in delta_vecs:
                            # print(r_state, delta_vec)
                            neighbor_state = [r_state[i] + delta_vec[i] for i in range(len(delta_vec))]
                            if plant is not None and (self.dynamics is 'MC' or self.dynamics is 'ABS'
                                                      or self.dynamics is 'Quadrotor' or self.dynamics is 'OtherBenchDisc1'
                                                      or self.dynamics is 'InvPendulumD' or self.dynamics is 'VertCAS'):
                                neighbor_trajectory = plant.getSimulations(states=[neighbor_state], do_not_parse=True)[0]
                            else:
                                # Added below r_time statement to take care of the warning - haven't tested it though.
                                # It has been working without this. If it causes an issue, comment it.
                                r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
                                neighbor_trajectory = generateTrajectories(self.dynamics, [neighbor_state], r_time, plant)[0]
                            trajectories.append(neighbor_trajectory)
                    self.trajectories = trajectories
                else:
                    self.trajectories = r_trajectories
                return self.trajectories

    def generateTrajectories(self, scaling=0.01, samples=None, r_states=None):
        return self.generateTrajectories4GradRun(scaling=scaling, samples=samples, r_states=r_states)
        # if self.dynamics is 'MC':
        #     dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/mountain_car/' + 'sig16x16.yml'
        #     dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
        #     plant = Plant('MC', dnn_controller_obj, None, self.steps)
        #     if r_states is not None:
        #         trajectories = plant.getSimulations(states=r_states)
        #         return trajectories
        #     else:
        #         if samples is None:
        #             states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        #             self.trajectories = plant.getSimulations(states=states)
        #             return self.trajectories
        #         else:
        #             states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        #             trajectories = plant.getSimulations(states=states)
        #             return trajectories
        # elif self.dynamics is 'Quadrotor':
        #     dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/quadrotor/' + 'tanh20x20.yml'
        #     dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
        #     plant = Plant('Quadrotor', dnn_controller_obj, None, self.steps)
        #     if r_states is not None:
        #         trajectories = plant.getSimulations(states=r_states)
        #         return trajectories
        #     else:
        #         if samples is None:
        #             states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        #             self.trajectories = plant.getSimulations(states=states)
        #             return self.trajectories
        #         else:
        #             states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        #             trajectories = plant.getSimulations(states=states)
        #             return trajectories
        # elif self.dynamics is 'ABS':
        #     dnn_cntrl_fname = '/home/manishg/Research/cps-falsification/verisig/examples/ABS/' + 'controller.yml'
        #     dnn_tf_fname = '/home/manishg/Research/cps-falsification/verisig/examples/ABS/' + 'transform.yml'
        #     dnn_controller_obj = DnnController(dnn_cntrl_fname, self.dimensions)
        #     dnn_transform_obj = DnnController(dnn_tf_fname, 2)
        #     plant = Plant('ABS', dnn_controller_obj, dnn_transform_obj, self.steps)
        #     if r_states is not None:
        #         trajectories = plant.getSimulations(states=r_states)
        #         return trajectories
        #     else:
        #         if samples is None:
        #             states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        #             self.trajectories = plant.getSimulations(states=states)
        #             return self.trajectories
        #         else:
        #             states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        #             trajectories = plant.getSimulations(states=states)
        #             return trajectories
        # elif self.dynamics is 'VehiclePlatoon' or self.dynamics is 'SpikingNeuron':
        #     plant = Plant(self.dynamics, None, None, self.steps)
        #     if r_states is not None:
        #         trajectories = plant.getSimulations(states=r_states)
        #         return trajectories
        #     else:
        #         if samples is None:
        #             states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        #             self.trajectories = plant.getSimulations(states=states)
        #             return self.trajectories
        #         else:
        #             states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        #             trajectories = plant.getSimulations(states=states)
        #             return trajectories
        # # elif self.dynamics is 'AeroBench':
        # #     aeroBenchSimObj = AeroBenchSim(dimensions=self.dimensions, timeStep=self.timeStep, steps=self.steps)
        # #     if r_states is not None:
        # #         trajectories = aeroBenchSimObj.getSimulations(states=r_states)
        # #         return trajectories
        # #     else:
        # #         if samples is None:
        # #             states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)
        # #             self.trajectories = aeroBenchSimObj.getSimulations(states=states)
        # #             return self.trajectories
        # #         else:
        # #             states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        # #             trajectories = aeroBenchSimObj.getSimulations(states=states)
        # #             return trajectories
        #
        # if r_states is not None:
        #     r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
        #     trajectories = generateTrajectories(self.dynamics, r_states, r_time)
        #     return trajectories
        # else:
        #     if samples is None:
        #         self.storeTrajectories()
        #         return self.trajectories
        #         # idx = random.randint(0, self.samples - 1)
        #         # print("random state: {}".format(self.states[idx]))
        #     else:
        #         r_states = generateRandomStates(samples, self.lowerBoundArray, self.upperBoundArray)
        #         r_time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
        #         trajectories = generateTrajectories(self.dynamics, r_states, r_time)
        #         return trajectories

    def showTrajectories(self, trajectories=None, xindex=0, yindex=1, dimwise=False):
        if trajectories is None:
            plotTrajectories(self.trajectories, xindex=xindex, yindex=yindex, dimwise=dimwise)
        else:
            plotTrajectories(trajectories, xindex=xindex, yindex=yindex, dimwise=dimwise)

    def storeStatesRandomSample(self):
        self.states = generateRandomStates(self.samples, self.lowerBoundArray, self.upperBoundArray)

    def storeTrajectories(self):

        if self.states == [] :
            self.storeStatesRandomSample()

        assert not (self.states == [])
        # assert self.lowerBoundArray is not [] and self.upperBoundArray is not []
        self.time = np.linspace(0, self.timeStep * self.steps, self.steps + 1)
        self.trajectories = generateTrajectories(self.dynamics, self.states, self.time)

    def dumpTrajectoriesStates(self, eval_var):
        f_name = "../models_trajs/states_"
        f_name = f_name + eval_var + "_"
        f_name = f_name + self.dynamics
        f_name = f_name + ".txt"
        if path.exists(f_name):
            os.remove(f_name)
        states_f = open(f_name, "w")
        for idx in range(len(self.states)):
            states_f.write(str(self.states[idx]))
            states_f.write("\n")
        states_f.close()


# config1 = configuration()
# time step default = 0.01, number of steps default = 100
# dimensions default = 2, number of sample default = 50

# config1.setSteps(100)
# config1.setSamples(10)

# config1.setDynamics('Vanderpol')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([50.0, 50.0])

# config1.setDynamics('Brussellator')
# config1.setLowerBound([1.0, 1.0])
# config1.setUpperBound([2.0, 2.0])

# config1.setDynamics('Lorentz')
# config1.setLowerBound([1.0, 1.0, 1.0])
# config1.setUpperBound([10.0, 10.0, 10.0])

# for i in range(0, 1):

# config1.storeTrajectories()
# config1.rawAnalyze()
# print (config1.eigenElbow)
# print (config1.noProminetEigVals)

# for j in range(0, config1.noProminetEigVals):
# print (config1.sortedVectors[j])
# config1.showLogEigenValues()
# config1.showTrajs()
