import mne
import numpy as np
import random
import os
from copy import deepcopy
import mne
import pickle as pkl
from tqdm import tqdm
import colorednoise as cn
import pickle as pkl
from joblib import Parallel, delayed
from ..util import *

def run_simulations(pth_fwd, n_simulations=10000, n_sources=(1, 5), extents=(2, 3), 
    amplitudes=(5, 10), shape='gaussian', durOfTrial=1, sampleFreq=100, 
    regionGrowing=True, n_jobs=-1, return_raw_data=False, return_single_epoch=True):
    ''' A wrapper function for the core function "simulate_source" which
    calculates simulations multiple times. 
    Parameters:
    -----------
    pth_fwd : str, path of the forward model folder containing forward model
    files n_simulations : int, number of simulations to perform. 100,000 perform
        great, 10,000 are fine for testing. 
    parallel : bool, perform simulations in parallel (can be faster) or sequentially 
    n_jobs : int, number of jobs to run in parallel, -1 utilizes all cores
    return_raw_data : bool, if True the function returns a list of 
        mne.SourceEstimate objects, otherwise it returns raw data

    <for the rest see function "simulate_source"> 
    
    Parameters:
    -----------
    sources : list, list of simulations containing either mne.SourceEstimate 
        objects or raw arrays (see <return_raw_data> argument)
    '''

    if not pth_fwd.endswith('/'):
        pth_fwd += '/'
    # Load neighbor matrix
    fwd_file = os.listdir(pth_fwd)[np.where(['-fwd.fif' in list_of_files 
        for  list_of_files in os.listdir(pth_fwd)])[0][0]]

    fwd = mne.read_forward_solution(pth_fwd + fwd_file, verbose=0)
    tris_lr = [fwd['src'][0]['use_tris'], fwd['src'][1]['use_tris']]
    neighbors = get_triangle_neighbors(tris_lr)
    # Load dipole positions in
    with open(pth_fwd + '/pos.pkl', 'rb') as file:  
        pos = pkl.load(file)[0]
    
    # perform simulations
    settings = {'n_sources':n_sources,
                'extents': extents, 
                'amplitudes': amplitudes,
                'shape': shape, 
                'durOfTrial': durOfTrial,
                'sampleFreq': sampleFreq,
                'regionGrowing': regionGrowing
                }

    print(f'\nRun {n_simulations} simulations...')

    sources = np.stack(Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(simulate_source)(pos, neighbors, **settings) 
        for i in tqdm(range(n_simulations))))

    if not return_raw_data:
        source_vectors = np.stack([source[0] for source in sources], axis=0)
        has_temporal_dimension = len(np.squeeze(source_vectors).shape) == 3
        if return_single_epoch and not has_temporal_dimension:
            print(f'\nConvert simulations to a single instance of mne.SourceEstimate...')
            sources = source_to_sourceEstimate(source_vectors, pth_fwd, sfreq=sampleFreq, simulationInfo=sources[0][1]) 
        else:
            print(f'\nConvert simulations to instances of mne.SourceEstimate...')
            sources = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(source_to_sourceEstimate)(source[0], pth_fwd, sfreq=sampleFreq, simulationInfo=source[1]) 
                for source in tqdm(sources))
    else:
        sources = np.stack([sources[i][0] for i in range(n_simulations)], axis=0)
            
    return sources

def simulate_source(pos, neighbors, n_sources=(1, 5), extents=(2, 3), amplitudes=(5, 10),
    shape='gaussian', durOfTrial=1, sampleFreq=100, regionGrowing=True):
    ''' Returns a vector containing the dipole currents. Requires only a dipole 
    position list and the simulation settings.

    Parameters:
    -----------
    pos : numpy.ndarray, (n_dipoles x 3), list of dipole positions.
    n_sources : int/tuple/list, number of sources. Can be a single number or a 
        list of two numbers specifying a range.
    regionGrowing : bool, whether to use region growing. If True, please supply
        also the neighbors to the settings.
    neighbors : list, a list containing all the (triangle-) neighbors for each 
        dipole. Can be calculated using "get_triangle_neighbors"
    extents : int/float/tuple/list, size of sources. If regionGrowing==True this 
        specifies the neighborhood order (see Grova et al., 2006), otherwise the diameter in mm. Can be a single number or a 
        list of two numbers specifying a range.
    amplitudes : int/float/tuple/list, the current of the source in nAm
    shape : str, How the amplitudes evolve over space. Can be 'gaussian' or 'flat' (i.e. uniform).
    durOfTrial : int/float, specifies the duration of a trial.
    sampleFreq : int, specifies the sample frequency of the data.
    Return:
    -------
    source : numpy.ndarray, (n_dipoles x n_timepoints), the simulated source signal
    simSettings : dict, specifications about the source.
    Grova, C., Daunizeau, J., Lina, J. M., B??nar, C. G., Benali, H., & Gotman, J. (2006). Evaluation of EEG localization methods using realistic simulations of interictal spikes. Neuroimage, 29(3), 734-753.
    '''
    
    # Handle input

    # Amplitudes come in nAm
    if isinstance(amplitudes, (list, tuple)):
        amplitudes = [amp* 1e-9  for amp in amplitudes] 
    else:
        amplitudes *= 1e-9

    if isinstance(extents, (list, tuple)):
        if np.max(extents) > 15 and regionGrowing:
            print(f'WARNING: When region growing is selected, extent refers to the neighborhood order. Your order goes up to {np.max(extents)}, but should be max at 10.')
            return
    else:
        if extents > 15 and regionGrowing:
            print(f'WARNING: When region growing is selected, extent refers to the neighborhood order. Your order is set to {np.max(extents)}, but should be max at 10.')
            return


    if durOfTrial > 0:
        if durOfTrial < 0.5 :
            print(f'durOfTrial should be either 0 or at least 0.5 seconds!')
            return
        
        signalLen = int(sampleFreq*durOfTrial)
        pulselen = sampleFreq/10
        pulse = get_pulse(pulselen)
        signal = np.zeros((signalLen))
        start = int(np.floor((signalLen - pulselen) / 2))
        end = int(np.ceil((signalLen - pulselen) / 2))
        signal[start:-end] = pulse
        signal /= np.max(signal)
    else:  # else its a single instance
        sampleFreq = 0
        signal = 1
    
    ###########################################
    # Select ranges and prepare some variables:
    sourceMask = np.zeros((pos.shape[0]))
    # If n_sources is a range:
    if isinstance(n_sources, (tuple, list)):
        n_sources = random.randrange(*n_sources)
  
    if isinstance(extents, (tuple, list)):
        extents = [random.randrange(*extents) for _ in range(n_sources)]
    else:
        extents = [extents for _ in range(n_sources)]

    if isinstance(amplitudes, (tuple, list)):
        amplitudes = [random.uniform(*amplitudes) for _ in range(n_sources)]
    else:
        amplitudes = [amplitudes for _ in range(n_sources)]
    
    src_centers = np.random.choice(np.arange(pos.shape[0]), \
        n_sources, replace=False)

    
    source = np.zeros((pos.shape[0]))
    
    ##############################################
    
    for i, src_center in enumerate(src_centers):
        # Smoothing and amplitude assignment
        if regionGrowing:
            d = get_n_order_indices(extents[i], src_center, neighbors)
            dists = np.empty((pos.shape[0]))
            dists[:] = np.inf
            dists[d] = np.sqrt(np.sum((pos - pos[src_center, :])**2, axis=1))[d]
        else:
            dists = np.sqrt(np.sum((pos - pos[src_center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]


        if shape == 'gaussian':
            # sd = extents[i]/2  # <-This does not work when extents can also be neighborhood orders
            sd = np.max(dists[d]) / 2  # <- works better
            source[:] += gaussian(dists, 0, sd) * amplitudes[i]
        elif shape == 'flat':
            source[d] += amplitudes[i]
        else:
            raise(BaseException, "shape must be of type >string< and be either >gaussian< or >flat<.")
        sourceMask[d] = 1

    # if durOfTrial > 0:
    n = np.clip(int(sampleFreq * durOfTrial), a_min=1, a_max=None)
    sourceOverTime = repeat_newcol(source, n)
    source = np.squeeze(sourceOverTime * signal)
    if len(source.shape) == 1:
        source = np.expand_dims(source, axis=1)
    
    # Prepare informative dictionary that entails all infos on how the simulation was created.
    simSettings = dict(scr_center_indices=src_centers, amplitudes=amplitudes, extents=extents, 
        shape=shape, sourceMask=sourceMask, regionGrowing=regionGrowing, durOfTrial=durOfTrial,
        sampleFreq=sampleFreq)

    return source, simSettings

def get_pulse(x):
    ''' Returns a pulse of length x'''
    freq = (1/x) / 2
    time = np.arange(x)

    signal = np.sin(2*np.pi*freq*time)
    return signal

def repeat_newcol(x, n):
    ''' Repeat a list/numpy.ndarray x in n columns.'''
    out = np.zeros((len(x), n))
    for i in range(n):
        out[:,  i] = x
    return np.squeeze(out)

def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    '''
    if order == 0:
        return pick_idx
    flatten = lambda t: [item for sublist in t for item in sublist]

    current_indices = [pick_idx]
    for cnt in range(order):
        # current_indices = list(np.array( current_indices ).flatten())
        new_indices = [neighbors[i] for i in current_indices]
        new_indices = flatten( new_indices )
        current_indices.extend(new_indices)
    return current_indices

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_triangle_neighbors(tris_lr):
    if not np.all(np.unique(tris_lr[0]) == np.arange(len(np.unique(tris_lr[0])))):
        for hem in range(2):
            old_indices = np.sort(np.unique(tris_lr[hem]))
            new_indices = np.arange(len(old_indices))
            for old_idx, new_idx in zip(old_indices, new_indices):
                tris_lr[hem][tris_lr[hem] == old_idx] = new_idx

        print('indices were weird - fixed them.')
    numberOfDipoles = len(np.unique(tris_lr[0])) + len(np.unique(tris_lr[1]))
    neighbors = [list() for _ in range(numberOfDipoles)]
    # correct right-hemisphere triangles
    tris_lr_adjusted = deepcopy(tris_lr)
    # the right hemisphere indices start at zero, we need to offset them to start where left hemisphere indices end.
    tris_lr_adjusted[1] += int(numberOfDipoles/2)
    # left and right hemisphere
    for hem in range(2):
        for idx in range(numberOfDipoles):
            # Find the indices of the triangles where our current dipole idx is part of
            trianglesOfIndex = tris_lr_adjusted[hem][np.where(tris_lr_adjusted[hem] == idx)[0], :]
            for tri in trianglesOfIndex:
                neighbors[idx].extend(tri)
                # Remove self-index (otherwise neighbors[idx] is its own neighbor)
                neighbors[idx] = list(filter(lambda a: a != idx, neighbors[idx]))
            # Remove duplicates
            neighbors[idx] = list(np.unique(neighbors[idx]))
            # print(f'idx {idx} found in triangles: {neighbors[idx]}') 
    return neighbors

def add_noise(x, snr, beta=0):
    x = np.squeeze(np.array(x))

    if len(x.shape) == 1:
        n_samples = np.clip(len(x), a_min=2, a_max=None)
        noise = cn.powerlaw_psd_gaussian(beta, n_samples)[:len(x)]
    else:
        n_samples = x.shape
        noise = cn.powerlaw_psd_gaussian(beta, n_samples)
    
    if len(noise) == 1:
        rms_noise = noise
    else:
        noise -= np.mean(noise)
        rms_noise = rms(noise)

    if len(x) == 1:
        rms_x = x
    else:
        rms_x = rms(x)
    
    rms_noise = rms(noise)
    noise_scaler = rms_x / (rms_noise*snr)
    return x + noise*noise_scaler

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def create_eeg_helper(eeg_sample, n_trials, snr, beta):
    if type(snr) == tuple or type(snr) == list:
        snr = random.uniform(*snr)
    # eeg_sample = np.repeat(eeg_sample, n_trials, axis=0)
    eeg_sample = np.repeat(np.expand_dims(eeg_sample, 0), n_trials, axis=0)

    # noise_trial = np.stack([add_noise(eeg_sample, snr, beta) for trial in range(n_trials)], axis=0)
    noise_trial = add_noise(eeg_sample, snr, beta)
    
    return noise_trial


def create_eeg(sourceEstimates, pth_fwd, snr=2, n_trials=20, beta=1, n_jobs=-1,
    return_raw_data=False, return_single_epoch=True):
    ''' Create EEG of specified number of trials based on sources and some SNR.
    Parameters:
    -----------
    sourceEstimates : list, list containing mne.SourceEstimate objects
    pth_fwd : str, path to the forward model files
    snr : tuple/list/float, desired signal to noise ratio within individual 
        trials. Can be a list or tuple of two floats specifying a range.
    n_trials : int, number of simulated trials
    beta : float, determines the frequency spectrum of the noise added 
        to the signal: power = (1/f)^beta. 
        0 will yield white noise, 
        1 will yield pink noise (1/f spectrum)
    n_jobs : int, Number of jobs to run in parallel. 
        -1 will utilize all cores.
    return_raw_data : bool, if True the function returns a list of 
        mne.SourceEstimate objects, otherwise it returns raw data
    Return:
    -------
    epochs : list, list of either mne.Epochs objects or list of raw EEG 
        data (see argument <return_raw_data> to change output).
    '''
    # Unpack the source data from the SourceEstimate objects
    if type(sourceEstimates) == mne.source_estimate.SourceEstimate:
        sources = np.transpose(sourceEstimates.data)
        sfreq = sourceEstimates.simulationInfo['sampleFreq']
        n_timepoints = 1
    elif type(sourceEstimates) == list:
        sources = np.stack([se.data for se in sourceEstimates], axis=0)
        sfreq = sourceEstimates[0].simulationInfo['sampleFreq']
        n_timepoints = sources.shape[-1]
    elif type(sourceEstimates) == np.ndarray:
        sources = np.squeeze(sourceEstimates)
        if len(sources.shape) == 2:
            sources = np.expand_dims(sources, axis=-1)
        sfreq = 1
        print(f'sources.shape={sources.shape}')
        n_timepoints = sources.shape[-1]
    else:
        msg = f'sourceEstimates must be of type <list> or <mne.source_estimate.SourceEstimate> but is of type <{type(sourceEstimates)}>'
        raise ValueError(msg)

    # Load some forward model objects
    leadfield = load_leadfield(pth_fwd)
    info = load_info(pth_fwd)
    info['sfreq'] = sfreq
    
    n_samples = len(sources)
    n_elec = leadfield.shape[0]
    

    eeg_clean = np.stack([np.matmul(leadfield, y) for y in sources], axis=0)

    eeg_trials_noisy = np.zeros((n_samples, n_trials, n_elec, n_timepoints))

    print(f'\nCreate EEG trials with noise...')
    eeg_trials_noisy = np.stack(Parallel(n_jobs=n_jobs, backend='loky')
        (delayed(create_eeg_helper)(eeg_clean[sample], n_trials, snr, beta) 
        for sample in tqdm(range(n_samples))), axis=0)
    
    if n_trials == 1 and len(eeg_trials_noisy.shape) == 2:
        # Add empty dimension to contain the single trial
        eeg_trials_noisy = np.expand_dims(eeg_trials_noisy, axis=1)

    
    if len(eeg_trials_noisy.shape) == 3:
        eeg_trials_noisy = np.expand_dims(eeg_trials_noisy, axis=-1)
        
    if eeg_trials_noisy.shape[2] != n_elec:
        eeg_trials_noisy = np.swapaxes(eeg_trials_noisy, 1, 2)

    if not return_raw_data:
        if return_single_epoch:
            print(f'\nConvert EEG matrices to a single instance of mne.Epochs...')
            print(f'eeg_trials_noisy.shape={eeg_trials_noisy.shape}')
            ERP_samples_noisy = np.mean(eeg_trials_noisy, axis=1)
            epochs = eeg_to_Epochs(ERP_samples_noisy, pth_fwd, info=info)

        else:
            print(f'\nConvert EEG matrices to instances of mne.Epochs...')
            epochs = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(eeg_to_Epochs)(sample, pth_fwd, info=info) 
                for sample in tqdm(eeg_trials_noisy))
    else:
        epochs = eeg_trials_noisy

    return epochs

# def create_eeg(sources, pth_fwd, snr=1, n_trials=20, beta=0):
#     ''' Create EEG of specified number of trials based on sources and some SNR.'''
#     with open(pth_fwd + '/leadfield.pkl', 'rb') as file:
#         leadfield = pkl.load(file)[0]
#     n_samples = len(sources)
#     n_elec = leadfield.shape[0]
#     n_timepoints = sources[0][0].shape[1]

#     eeg_clean = np.stack([np.matmul(leadfield, y[0]) for y in sources], axis=0)



#     eeg_trials_noisy = np.zeros((n_samples, n_trials, n_elec, n_timepoints))
    
#     for sample in tqdm(range(n_samples)):
#         noise_trial = np.stack([add_noise(eeg_clean[sample], snr, beta) for trial in range(n_trials)], axis=0)
#         if len(noise_trial.shape) == 2:
#             noise_trial = np.expand_dims(noise_trial, axis=-1)

#         eeg_trials_noisy[sample, :, :, :] = noise_trial

#     return eeg_trials_noisy
