import numpy as np
import pandas as pd
import fastdtw

import matplotlib.pyplot as plt
import seaborn as sns
import librosa, librosa.display
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ms2samples(ms, sampling_rate_kHz=20):
    '''
    from ms to number of samples
    '''
    return ms*sampling_rate_kHz

def samples2ms(samples, sampling_rate_kHz=20):
    '''
    from number of samples to ms
    '''
    return samples/sampling_rate_kHz


### functions for spectrograms ###
def get_spectral_activity(spec_files, audioevt_properties, spectral_window, spectral_bin_size=80):
    '''
    Returns a 2D numpy array of spectral activity within window.
    The activity is 3D in nature (renditions, channels, time), but is reshaped into 2D (renditions, channels*time)
    '''
    spectral_total = []
    for i, syllable in enumerate(audioevt_properties['syllable_start']):
        file_num = audioevt_properties['file_num'][i]
    
        window_start = int(syllable + spectral_window[0])
        window_end = int(syllable + spectral_window[1])

        binned_window_start = window_start//spectral_bin_size
        binned_window_end = window_end//spectral_bin_size
        # if binned_window_end - binned_window_start != (spectral_window[1] - spectral_window[0])//spectral_bin_size + 1:
        #     binned_window_end = binned_window_end - (binned_window_end - binned_window_start - (spectral_window[1] - spectral_window[0])//spectral_bin_size - 1)
    
        idx_within_window = range(binned_window_start, binned_window_end)
    
        
        spectral_per_syllable = spec_files[file_num-1][:,idx_within_window]
        spectral_total.append(spectral_per_syllable.flatten())
    
    spectral_total = np.array(spectral_total)
    
    return spectral_total
    
def get_spectral_activity_with_index(spectral_activity):
    '''
    Returns spectral activity with index. Index is required for computing time warped distances
    '''
    return np.concatenate((np.arange(spectral_activity.shape[0]).reshape((spectral_activity.shape[0],1)),spectral_activity), axis=1)
   
    
def plot_spectrogram(spectral_activity, title='Syllable Spectrogram', sr=20000, hop_length=80):
    '''
    Draws a spectrogram of the spectral activity
    '''
    fig, ax = plt.subplots(1,1, figsize=(8, 5))
    librosa.display.specshow(spectral_activity[0].reshape((128,-1)), x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar();
    plt.tight_layout();
    ax.set_xlabel('Time (s)')
    plt.title(title);


### functions for neural activities ###
def channel_activity(start_time, window, channel_id, neural_properties, bin_size = 1):
    '''
    Returns a vector of how many spikes happened per bin (of samples) for a given syllable and channel_id pair
    '''
    window_start = start_time + window[0]
    window_end = start_time + window[1]
    num_bins = int(np.ceil((window[1]-window[0])/bin_size))

    spike_times = neural_properties[neural_properties['channel_id']==channel_id]['spikes'].values[0]
    idx_within_window = (spike_times >= window_start) & (spike_times < window_end)

    spike_times_within_window = spike_times[idx_within_window]
    spike_times_within_window = spike_times_within_window - window_start  ##make timing relative

    binned_times = spike_times_within_window//bin_size  ## each spike time represented in bin number
    uniques, counts = np.unique(binned_times, return_counts=True)
    activity = np.zeros(num_bins)
    for i, u in enumerate(uniques):
        activity[u] = counts[i]

    return activity
    

def get_neural_properties_list(audioevt_fsource, spk_fsource, spk_cluster_info, spk_times, spk_cluster):
    '''
    Returns a list of neural_properties.
    A 'neural_properties' is a dataframe that has the firing timepoints of each neuropixel channel.
    '''
    neural_properties_list = []
    num_files = np.unique(audioevt_fsource).shape[0]
    
    for i in (range(num_files)):
        neural_properties = pd.DataFrame(data=spk_cluster_info['id'].unique(), columns=['channel_id'])
        neural_properties['file_num'] = np.repeat(i+1, neural_properties.shape[0])
        spk_i_idx = np.isin(spk_fsource, [i+1])
        spk_i_times = spk_times[spk_i_idx]
        spk_i_clusters = spk_cluster[spk_i_idx]
        
        spikes_by_channels = []
        for id in neural_properties['channel_id']:
            spikes = spk_i_times[spk_i_clusters == id]
            spikes_by_channels.append(spikes)
        neural_properties['spikes'] = spikes_by_channels
        neural_properties_list.append(neural_properties)
        
    return neural_properties_list
    

def get_neural_activity(neural_properties_list, audioevt_properties, channels_of_interest, neural_window, neural_bin_size=600):
    '''
    Returns a numpy array of neural activity within window
    '''
    neural_total = []
    for i, syllable in enumerate(audioevt_properties['syllable_start']):
        file_num = audioevt_properties['file_num'][i]
        neural_properties = neural_properties_list[file_num-1]
        neural_per_syllable = []
        for channel in channels_of_interest:
            activity = channel_activity(syllable, neural_window, channel, neural_properties, bin_size=neural_bin_size)
            neural_per_syllable.append(activity)
    
        neural_per_syllable = np.array(neural_per_syllable)
        neural_total.append(neural_per_syllable.flatten())
    
    neural_total = np.array(neural_total)
    
    return neural_total

def get_neural_activity_with_index(neural_activity):
    '''
    Returns neural activity with index. Index is required for computing time warped distances
    '''
    return np.concatenate((np.arange(neural_activity.shape[0]).reshape((neural_activity.shape[0],1)),neural_activity), axis=1)


def plot_raster(neural_data, rendition_number, channels_of_interest, neural_bin_size, x_tick_every=50, y_tick_every=30, figsize=(10,10)):
    '''
    Draws a raster plot, given neural rendtion and channels of interest
    '''
    neural_rendition = neural_data[rendition_number].reshape(len(channels_of_interest),-1)
    num_channels, num_timepoints = neural_rendition.shape
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.set_xticks(np.arange(0,num_timepoints,x_tick_every))
    if neural_bin_size is not None and neural_bin_size>0:
        ax.set_xticklabels(np.arange(0,num_timepoints,x_tick_every)/samples2ms(neural_bin_size))
        ax.set_xlabel('Time (ms)')
    else:
        ax.set_xlabel('Timepoints')
    ax.set_yticks(np.arange(0,num_channels, y_tick_every))
    ax.set_yticklabels(channels_of_interest[::y_tick_every])
    ax.set_ylabel('Channel id')
    im = ax.imshow(neural_rendition);
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, ticks=np.arange(neural_rendition.min(),neural_rendition.max()+1))
    
    
### functions for DTW ###
def path_for_pair(sylA, sylB, nrows=128):
    '''
    Given two syllables A and B, this function computes the time warped paths between the elements of the syllables
    '''
    if len(sylA.shape) < 2 :
        sylA = sylA.reshape((nrows,-1))
    if len(sylB.shape) < 2 :
        sylB = sylB.reshape((nrows,-1))

    _, pth = fastdtw.fastdtw(np.transpose(sylA), np.transpose(sylB), dist=1)
    return pth
    
    
def get_syllable_pairs(audioevt_properties, spectral_activity):
    '''
    Computes the time warped paths between syllable rendition pairs, for every pair.
    Could be a memory hog, might want to make it lighter
    '''
    syllable_path_pairs = {}
    # print('Computing DTW, and pairing syllables')
    for i in audioevt_properties['evt_id']:
        for j in audioevt_properties['evt_id']:
            syllable_path_pairs[(i,j)] = path_for_pair(spectral_activity[i,:],spectral_activity[j,:])
            
    # print('Done with DTW')
    return syllable_path_pairs
