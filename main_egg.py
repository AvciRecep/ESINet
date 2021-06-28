import numpy as np
from ESINet.util import *
from ESINet.ann import *

### Import source data
def import_source():
    # Write a function to import voxelised dipole information of all cases/simulations
    return sources_sim # np array of nSim x nVoxels x 1

def import_simulated_data():
    # Write a function to import simulation results of all cases/simulations
    return egg_sim # np array of nSim x nNodes x 1

if __name__ == '__main__':
    # Call import functions
    sources_sim = import_source()
    egg_sim = import_simulated_data()

    # Initialize the artificial neural network model
    input_dim = ? #total number of voxels
    output_dim = ? #number of selected nodes
    model = get_model(input_dim, output_dim)

    # Train the model
    model, history = train_model(model, sources_sim, egg_sim, delta=1)

    # Test the model
    source_predicted = predict(model, egg_sample, pth_fwd)
    # Modify 'predict' function not to use pth_fwd or leadfield.

    # Evaluate the prediction
