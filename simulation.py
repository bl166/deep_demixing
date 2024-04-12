import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy.core.numeric import full_like
import torch
import sys
import os
import pickle
import argparse
import time

from utils import *
from config import set_params

# ------ GENERATING DATASETS -------


def generate_dataset(
    graph, N_sims, simulation_model, simulation_params, 
    observation_model, observation_params
    ):

    sim_param = {k:v for k,v in simulation_params.items()}
    data = []
    for curr_sim in range(N_sims):
        
        sim_param['starting_node'] = simulation_params['starting_node'][curr_sim]
        sim_result = simulation_model(graph, sim_param)
        data_uncollapsed = simulation_to_numpy(sim_result)
        
        if 0:#curr_sim == 0:
            plt.figure()
            plt.plot(np.sum(data_uncollapsed, 0))
            plt.show()
            plt.savefig('blbl.png')

        observation_params['data_uncollapsed'] = data_uncollapsed
        observation_params['graph'] = graph

        target_data, data_aggregated  = observation_model(observation_params)

        data.append([target_data, data_aggregated])            

    adj = nx.to_numpy_matrix(graph)
    node_attributes = nx.get_node_attributes(graph, 'pos')
    dataset = {'adj' : adj, 'data' : data, 'attr' : node_attributes, 'graph' : graph}
    
    try:
        node_class = nx.get_node_attributes(graph, 'cls')
        dataset['cls'] = node_class
        print('graph with node class...')
    except:
        pass
    
    return dataset
    


def generate_scenario(
    graph, N_sims, simulation_model, simulation_params, 
    observation_model, observation_params, save_path
    ):

        dataset = generate_dataset(
            graph, N_sims, simulation_model, simulation_params, 
            observation_model, observation_params
            )

        filehandler = open(save_path, 'wb')
        pickle.dump(dataset, filehandler)
        filehandler.close()

        print(f"\nDataset created and saved to: {save_path}\n")




# ------ SIMULATION MODELS -------


def SIRS_simulation(G, params):

    """
    Returns the full evoulution of the spread.
    -- final_values: dict 
        - key - number of the day 
        - value - dict_per_day
    
    -- dict_per_day: dict   
        - key - node index
        - value - 0 / 1 (not infected / infected)
        
    """

    try:
        starting_node = params['starting_node']
        S2I_probs = params['S2I']
        I2R_probs = params['I2R']
        R2S_probs = params['R2S']
        N_steps = params['N_steps']
        N_second = params['N_second']
    
    except KeyError:
        print("There are missing arguments in the params dict! ")
        sys.exit(1)

    S2I, I2R, R2S = S2I_probs[0], I2R_probs[0], R2S_probs[0]
    if N_second is not None:
        S2I_1, I2R_1, R2S_1 = S2I_probs[1], I2R_probs[1], R2S_probs[1]

    # All node values at a certain time step will be stored in a dictionary called node_values
    # Initialize the value of the first infected node to 1
    node_values = {starting_node : 1}
    recovered_dict = {}

    # Initialize all the rest to 0
    for node in G.nodes():
        recovered_dict.update({node : 0})
        if node != starting_node:
            node_values.update({node : 0})

    # Store all node_values 
    all_node_values = {0 : copy.copy(node_values)}
    
    timestep = 0
    # Run the simulation N_steps times: 
    while (timestep < N_steps - 1):

        if (not N_second == None) and (timestep > N_second):
            S2I = S2I_1
            I2R = I2R_1 
            R2S = R2S_1

        # For every susceptible node, flip the coin for every of its infected neighbors (generate a random value)
        # => if at least one value is smaller than S2I, the current node is considered infected, starting next time step

        # For every infected node, flip the coin => it becomes susceptible with probability I2R
        next_values = {}
        current_active = 0

        for node in G.nodes():

            # If the node is susceptible:
            if node_values[node] == 0:

                infected_neighbors = count_infected_neighbors(G, node, node_values)
                # Now we flip the coin for every infected neighbor:
                infected = False 

                for i in range(0, infected_neighbors):
                    random_value = np.random.rand()
                    if random_value < S2I:
                        infected = True 
                        break 
                    
                if infected == True:
                    next_values.update({node : 1})
                else: 
                    next_values.update({node : 0})


            # If the node is recovered:
            if (node_values[node] == 0) and (recovered_dict[node] == 1):

                infected_neighbors = count_infected_neighbors(G, node, node_values)
                # Now we flip the coin for every infected neighbor:
                susceptible = False 

                for i in range(0, infected_neighbors):
                    random_value = np.random.rand()
                    if random_value < R2S:
                        susceptible = True 
                        break 
                
                next_values.update({node : 0})
                    
                if susceptible == True:
                    recovered_dict[node] = 0


            # If the node is infected:
            if node_values[node] == 1:

                current_active += 1

                # Flip the coin:
                recovered = False 

                random_value = np.random.rand()
                if random_value < I2R:
                    recovered = True
                
                if recovered == True:
                    next_values.update({node : 0})
                    recovered_dict[node] = 1
                else: 
                    next_values.update({node : 1})


        # Update node_values for the next time step and reset next_values
        node_values = copy.copy(next_values)
        next_values = {}
        all_node_values.update({timestep + 1 : copy.copy(node_values)})

        timestep += 1

    return all_node_values



def SIRSD_simulation(G, params):

    """
    Returns the full evoulution of the spread.
    -- final_values: dict 
        - key - number of the day 
        - value - dict_per_day
    
    -- dict_per_day: dict   
        - key - node index
        - value - 0 / 1 (not infected / infected)
        
    """

    try:
        starting_node = params['starting_node']
        S2I_probs = params['S2I']
        I2R_probs = params['I2R']
        R2S_probs = params['R2S']
        I2D_probs = params['I2D']
        N_steps = params['N_steps']
        N_second = params['N_second']
    
    except KeyError:
        print("There are missing arguments in the params dict! ")
        sys.exit(1)

    S2I, I2R, R2S, I2D = S2I_probs[0], I2R_probs[0], R2S_probs[0], I2D_probs[0]
    if N_second is not None:
        S2I_1, I2R_1, R2S_1, I2D_1 = S2I_probs[1], I2R_probs[1], R2S_probs[1], I2D_probs[1]

    # All node values at a certain time step will be stored in a dictionary called node_values
    # Initialize the value of the first infected node to 1
    node_values = {starting_node : 1}
    recovered_dict = {}
    dead_dict = {}

    # Initialize all the rest to 0
    for node in G.nodes():
        recovered_dict.update({node : 0})
        dead_dict.update({node : 0})
        if node != starting_node:
            node_values.update({node : 0})

    # Store all node_values 
    all_node_values = {0 : copy.copy(node_values)}
    
    timestep = 0
    # Run the simulation N_steps times: 
    while timestep < N_steps - 1:

        if (not N_second == None) and (timestep > N_second):
            S2I = S2I_1
            I2R = I2R_1 
            R2S = R2S_1
            I2D = I2D_1

        # For every susceptible node, flip the coin for every of its infected neighbors (generate a random value)
        # => if at least one value is grater than S2I, the current node is considered infected, starting next time step

        # For every infected node, flip the coin => it becomes susceptible with probability I2R
        next_values = {}
        
        for node in G.nodes():

            # If the node is susceptible:
            if node_values[node] == 0 and recovered_dict[node] == 0 and dead_dict[node] == 0:
                infected_neighbors = count_infected_neighbors(G, node, node_values)
                # Now we flip the coin for every infected neighbor:
                infected = False 

                for _ in range(0, infected_neighbors):
                    random_value = np.random.rand()
                    if random_value < S2I:
                        infected = True 
                        break 
                    
                if infected == True:
                    next_values.update({node : 1})
                else: 
                    next_values.update({node : 0})


            # If the node is recovered:
            if node_values[node] == 0 and recovered_dict[node] == 1:
                infected_neighbors = count_infected_neighbors(G, node, node_values)
                # Now we flip the coin for every infected neighbor:
                susceptible = False 

                for i in range(0, infected_neighbors):
                    random_value = np.random.rand()
                    if random_value < R2S:
                        susceptible = True 
                        break 
                
                next_values.update({node : 0})
                    
                if susceptible == True:
                    recovered_dict[node] = 0


            # If the node is dead:
            if node_values[node] == 0 and dead_dict[node] == 1:
                next_values.update({node : 0})
                

            # If the node is infected:
            if node_values[node] == 1:
                # Flip the coin for D state:
                dead = False 

                random_value = np.random.rand()
                if random_value < I2D:
                    dead = True

                if dead == True:
                    next_values.update({node : 0})
                    dead_dict[node] = 1
                else:
                    # Flip the coin for R state:
                    recovered = False 

                    random_value = np.random.rand()
                    if random_value < I2R:
                        recovered = True
                    
                    if recovered == True:
                        next_values.update({node : 0})
                        recovered_dict[node] = 1
                    else: 
                        next_values.update({node : 1})


        # Update node_values for the next time step and reset next_values
        node_values = copy.copy(next_values)
        next_values = {}
        all_node_values.update({timestep + 1 : copy.copy(node_values)})

        timestep += 1

    return all_node_values




# ------ Simulation - Observation Bridge -------

def simulation_to_numpy(sim_result):

    """
    Value at (n, t):
        * 0 - not infected
        * 1 - infected
    """

    T = len(sim_result)
    assert T > 0
    N = len(list(sim_result.values())[0])
    assert N > 0
    
    # Assign -1 to every entry, so that we can check if some are missing
    result_np = np.ones((N, T)) * (-1)

    node_inds = np.arange(0, N)
    day_inds = np.arange(0, T)

    for curr_node in node_inds:
        for curr_day in day_inds:

            result_np[curr_node, curr_day] = sim_result[curr_day][curr_node]
    
    assert np.min(result_np) >= 0

    return result_np



# ------ OBSERVATION MODELS -------


def last_day_observation(params):
    """
    full_evolution - np.array, shape [N x T]
    returns:
        - target data - full_evolution
        - observed data - np.array, shape [N x 1]

    """

    try:
        non_observable_nodes = params['non_observable_nodes']
        prediction = params['prediction']
        full_evolution = params['data_uncollapsed']
    except KeyError:
        print("There are missing arguments in the params dict! ")
        sys.exit(1)

    if non_observable_nodes is None:
        non_observable_nodes = []
    
    last_day = full_evolution[ : , -1].reshape((-1, 1)).copy()
    last_day[non_observable_nodes] = -1
    
    if prediction:
        # If this observation model is used for prediction, 
        # then the target data is only the last day, 
        # and the observed data is the full evolution minus the last day
        return last_day, full_evolution[ : , : -1]
    else:
        return full_evolution, last_day



def average_across_days_observation(params):
    """
    Returns the average of all fully observable nodes on observable days. 

    full_evolution - np.array, shape [N x T]
    returns:
        - target data - full_evolution
        - observed data - np.array, shape [N_observable x 1]

    """

    try:
        non_observable_nodes = params['non_observable_nodes']
        observable_days  = params['observable_days']
        full_evolution = params['data_uncollapsed']
    except KeyError:
        print("There are missing arguments in the params dict! ")
        sys.exit(1)

    if non_observable_nodes is None:
        non_observable_nodes = []
    if observable_days is None:
        T = full_evolution.shape[1]
        observable_days = [i for i in range(T)]
    
    average = np.mean(full_evolution[ : , observable_days], 1).reshape((-1, 1)).copy()
    average[non_observable_nodes] = -1 
    return full_evolution, average


def spread_score_0(full_evolution, graph, normalize=None):

    [N, T] = list(full_evolution.shape)

    # Initialize all weights to zero
    spreading_weights = np.zeros(N)

    for day in range(1, T):
        for node in graph.nodes():
            # If a node is infected today and wasn't the day before
            # ==>> split the 'penalty' (or 'reward', depends how you look at it)
            #      among his INFECTED neighbors 
            if full_evolution[node][day] == 1 and full_evolution[node][day - 1] == 0:
                infected_neighbors = get_infected_neighbors(graph, node, full_evolution[ : , day - 1])

                for neighbor in infected_neighbors:
                    spreading_weights[neighbor] += float(1 / len(infected_neighbors))
                    
    spreading_weights = spreading_weights.reshape(-1)
    if normalize is not None:
        spreading_weights = normalize(spreading_weights)                 
    return spreading_weights
    

def super_spreaders_observation(params):

    """
    Returns an array denoting which nodes were considered super-spreaders for the 
    given epidemic spread (if it's the binary case) or to 
    what extent they contributed to the spread (if it's the continuous case).
    This vector is considered the target variable.
    Depending on the case, the corresponding data can be altered. 
    For now, it's only the full evolution of the spread.
    full_evolution - np.array, shape [N x T]
    returns:
        - spreading_weights - N-element array 
    """

    try:
        binary = params['binary']
        threshold = params['threshold']
        full_evolution = params['data_uncollapsed']
        graph = params['graph']
        input_type = params['input_type']
    except KeyError:
        print("There are missing arguments in the params dict! ")
        sys.exit(1)
        
    spreading_weights = spread_score_0(full_evolution, graph)
    input_data = full_evolution
    # if input_type == '...':

    if binary:
        binarized = np.array(spreading_weights > threshold, dtype=int)
        return binarized, input_data
    else:
        return spreading_weights, input_data



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--scenario', type=int, default=-1)
parser.add_argument('--timesteps', type=int, default=20)
parser.add_argument('--simulations', type=int, default=1000)
parser.add_argument('--nodes', type=int, default=100)
parser.add_argument('--topology', type=str, default='RG')
parser.add_argument('--rg_radius', type=float, default=0.3)
parser.add_argument('--probability', type=float, default=None)#1e-4)


if __name__ == '__main__':

    args = parser.parse_args()
    curr_script_dir = os.path.dirname(os.path.realpath(__file__))
    exp_datasets_dir = os.path.join(curr_script_dir, "datasets")
    os.makedirs(exp_datasets_dir, exist_ok=True)

    # ====== EXP: CUSTOM SIMULATION/OBSERVATION MODELS ======
    # ====== EXP: DIFFERENT GRAPH TOPOLOGIES ======

    # ------ GRAPH SETUP -------
    LOAD_GRAPH = False
    load_dataset_path = None
    
    node_number = args.nodes
    graph_param, gen_graph = set_params(args.topology, n=node_number, rad=args.rg_radius, p=args.probability)

    # ------ SIMULATION PARAMETERS -------
    
    S2I_probs = [0.5, 0.15]
    I2R_probs = [0.05, 0.1]
    R2S_probs = [0.005, 0.01]
    I2D_probs = [0.1, 0.1]
    N_second = None

    N_timesteps = args.timesteps
    N_simulations = args.simulations
    seed = args.seed + N_simulations
    
    starting_node = np.random.RandomState(seed).randint(low=0, high=node_number, size=N_simulations)

    simulation_params = {
        'starting_node' : starting_node, 
        'S2I' : S2I_probs, 'I2R' : I2R_probs, 'R2S' : R2S_probs, 'I2D' : I2D_probs,
        'N_steps' : N_timesteps, 'N_second' : N_second
        }
    
    if LOAD_GRAPH:   
        graph = load_graph_from_dataset(load_dataset_path, node_number)        
    else:
        graph = gen_graph(**graph_param, seed=seed)

    
    # ------ OBSERVATION PARAMETERS -------
    non_observable_nodes = None
        
    observable_days = None

    super_spreader_factor = 0.02

    observation_params = {
        'non_observable_nodes' : non_observable_nodes, 
        'observable_days' : observable_days,
        'prediction' : False,
        'binary' : False,# True,
        'threshold' : super_spreader_factor * node_number,
        'input_type' : 'full_evolution'
        }

    if 'prediction' in observation_params: 
        if observation_params['prediction']:
            simulation_params['N_steps'] += 1
            

    mstring = f"({len(observable_days)})" if observable_days is not None else ""
    name_fix = f"{node_number}nodes_{N_timesteps}" + mstring + f"steps_{N_simulations}sims_{str(graph_param)}gparam_{args.topology}_{args.seed}seed"
   
    # ------ SCENARIO 1: SIRS + AVERAGE -------
    if args.scenario == 1 or args.scenario == -1:

        dataset_name = "SIRS_average_"+name_fix+".obj"
        save_path = os.path.join(exp_datasets_dir, dataset_name)
        if not os.path.exists(save_path):
            generate_scenario(
                graph, N_simulations, SIRS_simulation, simulation_params,
                average_across_days_observation, observation_params, save_path
                )

    # ------ SCENARIO 3: SIRSD + AVERAGE -------
    if args.scenario == 3 or args.scenario == -1:

        dataset_name = "SIRSD_average_"+name_fix+".obj"
        save_path = os.path.join(exp_datasets_dir, dataset_name)
        if not os.path.exists(save_path):
            generate_scenario(
                graph, N_simulations, SIRSD_simulation, simulation_params,
                average_across_days_observation, observation_params, save_path
                )
            
    # ------ SCENARIO 7: SIRS + SUPER SPREADERS -------
    if args.scenario == 7:

        dataset_name = f"SIRS_super_spreaders_full_evolution_spread{super_spreader_factor}_" + name_fix + ".obj"
        save_path = os.path.join(exp_datasets_dir, dataset_name)
        if 1:#ot os.path.exists(save_path):
            generate_scenario(
                graph, N_simulations, SIRS_simulation, simulation_params,
                super_spreaders_observation, observation_params, save_path
                )            
