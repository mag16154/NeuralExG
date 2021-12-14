import matplotlib
import numpy as np

import sys
sys.path.append('../core')
sys.path.append('../configuration-setup/')
sys.path.append('../core/rrtAlgorithms/src/')

from evaluation_isen import EvaluationInvSen
from NNConfiguration import NNConfiguration
from itertools import combinations
from configuration import configuration

if __name__ == '__main__':
    # evalObject = EvaluationInvSen(dynamics='OtherBenchC1')
    # evalObject.setIterCount(100)
    # dataObject = evalObject.setDataObject()
    # dest_traj = dataObject.generateTrajectories(r_states=[[1.33236118, 1.17698394]])[0]
    # time_step = 77
    #
    # print(dest_traj[0], dest_traj[time_step])
    # dest = dest_traj[time_step]
    # dests = [dest]
    # ref_traj = dataObject.generateTrajectories(r_states=[[0.67184992,1.04788799]])[0]
    # print(ref_traj[0], ref_traj[time_step])
    # evalObject.reachDestInvPaths(dests=dests, d_time_steps=[time_step], threshold=0.005, i_state=[ref_traj[0]],
    #                              correction_steps=[1], scaling_factors=[0.025])

    evalObject = EvaluationInvSen(dynamics='OtherBenchC2')
    evalObject.setIterCount(100)
    dataObject = evalObject.setDataObject()
    dest_traj = dataObject.generateTrajectories(r_states=[[0.74224485, 1.1416229]])[0]
    time_step = 71  # 127
    # time_step = 127

    print(dest_traj[0], dest_traj[time_step])
    # dest = dest_traj[time_step]
    # dests = [dest]
    dests = [[ 1.26119845, -0.21816047]]  # For 127
    ref_traj = dataObject.generateTrajectories(r_states=[[1.11786202, 1.18747043]])[0]
    print(ref_traj[0], ref_traj[time_step])
    evalObject.reachDestInvPaths(dests=dests, d_time_steps=[time_step], threshold=0.01, i_state=[ref_traj[0]],
                                 correction_steps=[1], scaling_factors=[0.02], adaptation_run=False)

    # evalObject = EvaluationInvSen(dynamics='OtherBenchC9')
    # evalObject.setIterCount(2000)
    # dataObject = evalObject.setDataObject()
    # dataObject.setSteps(150)
    # dest_traj = dataObject.generateTrajectories(r_states=[[0.53432163, -0.50705631, -0.10642957, 0.90469208]])[0]
    # time_step = 137
    #
    # print(dest_traj[0], dest_traj[time_step])
    # dest = dest_traj[time_step]
    # dests = [dest]
    # ref_traj = dataObject.generateTrajectories(r_states=[[0.9510333, -0.98411012, -0.29126114, 0.78761868]])[0]
    # print(ref_traj[0], ref_traj[time_step])
    # evalObject.reachDestInvPaths(dests=dests, d_time_steps=[time_step], threshold=0.005,
    #                              correction_steps=[1], scaling_factors=[0.02],
    #                              i_state=[ref_traj[0]], adaptation_run=False)

    # evalObject = EvaluationInvSen(dynamics='OtherBenchC9Tanh')
    # evalObject.setIterCount(500)
    # dataObject = evalObject.setDataObject()
    # dataObject.setSteps(250)
    #
    # dest_traj = dataObject.generateTrajectories(r_states=[[-0.90804677, -0.2088056, 0.54512056, -0.38022436]])[0]
    # time_step = 177
    #
    # print(dest_traj[0], dest_traj[time_step])
    # dest = dest_traj[time_step]
    # dests = [dest]
    # ref_traj = dataObject.generateTrajectories(r_states=[[-0.52187253, -0.54350975, 0.54205043, -0.20722052]])[0]
    # print(ref_traj[0], ref_traj[time_step])
    # evalObject.reachDestInvPaths(dests=dests, d_time_steps=[time_step], threshold=0.005,
    #                              correction_steps=[1], scaling_factors=[0.01], i_state=[ref_traj[0]],
    #                              adaptation_run=True)
