import numpy as np
import matplotlib.pyplot as plt
from ESINet.util import *
from ESINet.ann import *


### Simulate some source activity and EEG data
if True:
    sources_sim = run_simulations(pth_fwd, n_simulations=100, durOfTrial=0)
    a = sources_sim[0,:,0]
    ids = np.nonzero(a)
    print('++++++++++++')
    print(a[ids])
    print('++++++++++++')
    eeg_sim = create_eeg(sources_sim, pth_fwd)
    #a = eeg_sim[0,0,:,0]
    #print(a)
### Plot a simulated sample and corresponding EEG
if False:
    #%matplotlib qt
    sample = 0  # index of the simulation
    title = f'Simulation {sample}'
    # Topographic plot
    eeg_sim[sample].average().plot_topomap([0.5])
    # Source plot
    sources_sim.plot(hemi='both', initial_time=sample, surface='white', colormap='inferno', title=title, time_viewer=False)

### Load and train ConvDip with simulated data
if True:
    # Find out input and output dimensions based on the shape of the leadfield
    input_dim, output_dim = load_leadfield(pth_fwd).shape
    print('-------')
    print(input_dim)
    print(output_dim)
    print('-------')
    # Initialize the artificial neural network model
    model = get_model(input_dim, output_dim)
    # Train the model
    model, history = train_model(model, sources_sim, eeg_sim, delta=1)

### Evaluate ConvDip
if False:
    #%matplotlib qt
    # Load some files from the forward model
    leadfield = load_leadfield(pth_fwd)
    info = load_info(pth_fwd)

    # Simulate a brand new sample:
    sources_eval = run_simulations(pth_fwd, 1, durOfTrial=0)
    eeg_eval = create_eeg(sources_eval, pth_fwd)

    # Calculate the ERP (average across trials):
    eeg_sample = np.squeeze( eeg_eval )

    # Predict
    source_predicted = predict(model, eeg_sample, pth_fwd)

    # Visualize ground truth...
    #title = f'Ground Truth'
    #sources_eval.plot(hemi='both', initial_time=0.5, surface='white', colormap='inferno', title=title, time_viewer=False)

    # ... and prediction
    #title = f'ConvDip Prediction'
    #source_predicted.plot(hemi='both', initial_time=0.5, surface='white', colormap='inferno', title=title, time_viewer=False)

    # ... and the 'True' EEG topography
    #title = f'Simulated EEG'
    #eeg_eval[0].average().plot_topomap([0], title=title)
