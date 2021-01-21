import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from copy import deepcopy
from ..util import *


def get_model(input_dim, output_dim):
    input_shape = (None, input_dim)
    # Hyperparameters:
    activation_function = 'swish'
    
    model = keras.Sequential()

    for i in range(1):
        model.add(layers.Dense(units=128,
                            activation=activation_function))

    model.add(layers.Dense(output_dim, 
        activation=keras.layers.ReLU(max_value=1)))
    # model.add()
    model.build(input_shape=input_shape)
    model.summary()
    return model

def train_model(model, sources, eeg, batch_size=200, epochs=50, validation_split=0.1, loss=None, 
    optimizer=None, device=None, delta=1):
    ''' Train the neural network with simulated data. Parameters:
    -----------
    model : keras model to be trained (https://keras.io/api/models/model/)
    sources : list, simulated sources, returned from function "run_simulations"
    eeg  : numpy.ndarray, eeg data of the sources, returned from function
    "create_eeg" batch_size : int, how many samples are processed together.
        Lower if you run out of ram epochs : int, how often all samples are used for
    training validation_split : float between 0 and 1, proportion of samples to
    use for validation loss : keras loss function (https://keras.io/api/losses/)
    optimizer : keras optimizer (https://keras.io/api/optimizers/) device : str,
        device string to train on, could be "/GPU:0" or "/CPU:0". You can view your
        devices by typing:
        print(tensorflow.python.client.device_lib.device_lib.list_local_devices())
    delta : float between 0 and inf, the delta parameter for huber loss. 0
        yields mean absolute error - like error, inf yields mean squared error -
        like error 
    Return:
    -------
    model : keras model, the trained model history : keras history, the history
    of the training (loss over time)
    '''
    # Handle EEG input
    if type(eeg[0]) == mne.epochs.EpochsArray:
        eeg = np.stack([ep.get_data() for ep in eeg], axis=0)
        # Check if there is a temporal dimension

    if len(eeg.shape) == 4 and eeg.shape[-1] > 1:
        print(f'Simulations have a temporal dimension (i.e. more than a single time point). Please simulate data without a temporal dimension!\n Solution: When using the function <run_simulations> set durOftrial=0.')
        raise ValueError('eeg must contain data without temporal dimension. ')
    
    # Handle EEG input
    if type(sources[0]) == mne.source_estimate.SourceEstimate:
        sources = np.stack([source.data for source in sources], axis=0)
        # Check if there is a temporal dimension

    if len(eeg.shape) == 4 and eeg.shape[-1] > 1:
        print(f'Simulations have a temporal dimension (i.e. more than a single time point). Please simulate data without a temporal dimension!\n Solution: When using the function <run_simulations> set durOftrial=0.')
        raise ValueError('eeg must contain data without temporal dimension. ')

    # Extract data
    y = np.squeeze(sources)
    x = np.squeeze(np.mean(eeg, axis=1))
    # Prepare data
    # Scale sources
    y_scaled = np.stack([sample / np.max(sample) for sample in y])
    # Common average referencing for eeg
    x = np.stack([sample - np.mean(sample) for sample in x])
    # Scale EEG
    x_scaled = np.stack([sample / np.max(np.abs(sample)) for sample in x])
    
    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
        mode='min', verbose=1, patience=25, restore_best_weights=True)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
    if loss is None:
        loss = tf.keras.losses.Huber(delta=delta)
    
    model.compile(optimizer, loss)
    if device is None:
        history = model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                validation_split=validation_split, verbose = 2, callbacks=[es])
    else:
        with tf.device(device):
            history = model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, \
                validation_split=validation_split, verbose = 2, callbacks=[es])
    return model, history

def predict(model, EEG, pth_fwd, leadfield=None, dtype='raw', sfreq=100):
    ''' 
    Parameters:
    -----------
    model : keras model
    EEG : numpy.ndarray, shape (timepoints, electrodes), EEG data to infer sources from 
    leadfield : numpy.ndarray, the leadfiled matrix
    pth_fwd : str, path to forward model
    dtype : str, either of:
        'raw' : will return the source as a raw numpy array 
        'SourceEstimate' or 'SE' : will return a mne.SourceEstimate object
    
    Return:
    -------
    outsource : either numpy.ndarray (if dtype='raw') or mne.SourceEstimate instance
    '''
    if leadfield is None:
        leadfield = load_leadfield(pth_fwd)
    
    EEG = np.squeeze(np.array(EEG))

    if len(EEG.shape) == 1:
        EEG = np.expand_dims(EEG, axis=0)
    
    if EEG.shape[1] != leadfield.shape[0]:
        EEG = EEG.T
        


    # Prepare EEG to ensure common average reference and appropriate scaling
    EEG_prepd = deepcopy(EEG)
    for i in range(EEG.shape[0]):
        EEG_prepd[i, :] -= np.mean(EEG_prepd[i, :])
        EEG_prepd[i, :] /= np.max(np.abs(EEG_prepd[i, :]))
    
    # Predict using the model
    source_predicted = model.predict(EEG_prepd)
    # Scale ConvDips prediction
    source_predicted_scaled = np.squeeze(np.stack([solve_p(source_frame, EEG_frame, leadfield) for source_frame, EEG_frame in zip(source_predicted, EEG)], axis=0))
    
    if dtype == 'raw':
        outsource = np.squeeze(source_predicted_scaled)
    elif dtype == 'SourceEstimate' or dtype == 'SE':
        outsource = source_to_sourceEstimate(source_predicted_scaled, pth_fwd, sfreq=100)
    else:
        raise ValueError('dtype must be raw, SourceEstimate or SE.')
    
    predicted_source_estimate = source_to_sourceEstimate(outsource, pth_fwd, sfreq=1)
    return predicted_source_estimate

def solve_p(y_est, x_true, leadfield):
    
    # Check if y_est is just zeros:
    if np.max(y_est) == 0:
        return y_est
    y_est = np.squeeze(np.array(y_est))
    x_true = np.squeeze(np.array(x_true))
    # Get EEG from predicted source using leadfield
    x_est = np.matmul(leadfield, y_est)

    # optimize forward solution
    tol = 1e-100
    options = dict(maxiter=1000)
    # base scaling

    rms_est = np.mean(np.abs(x_est))
    rms_true = np.mean(np.abs(x_true))
    base_scaler = rms_true / rms_est

    opt = minimize_scalar(mse_opt, args=(leadfield, y_est* base_scaler, x_true), \
        bounds=(-1, 1), method='bounded', options=options, tol=tol)
    scaler = opt.x
    y_scaled = y_est * scaler * base_scaler
    return y_scaled

def mse_opt(scaler, leadfield, y_est, x_true):
    x_est = np.matmul(leadfield, y_est) 
    error = np.abs(pearsonr(x_true-x_est, x_true)[0])
    
    return error
