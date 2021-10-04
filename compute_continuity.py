from misc import *
import numpy as np
import pandas as pd
import pickle
import itertools
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as kNN


def emd_extension(u_vec, v_vec, sigma=1):
    '''
    Returns a distance (based on emd) between u_vec and v_vec
    u_vec, v_vec : postivie vectors (neural signals)
    sigma : ratio between d1(mass difference) and d2(vanilla emd)
    '''
    s_u = np.sum(u_vec)
    s_v = np.sum(v_vec)
    ## f1:vector with larger mass, f2:vector with smaller mass
    if s_u >= s_v:
        f1 = np.array(u_vec)
        f2 = np.array(v_vec)
        f1_sum = s_u
        f2_sum = s_v
    else:
        f1 = np.array(v_vec)
        f2 = np.array(u_vec)
        f1_sum = s_v
        f2_sum = s_u

    d1 = f1_sum - f2_sum
    
    if f1_sum==0.:
        f1 = np.ones_like(f1)*1e-99
        f1_sum = f1.sum()
    if f2_sum==0.:
        f2 = np.ones_like(f2)*1e-99
        f2_sum = f2.sum()

    f1_probs = f1/f1_sum
    f2_probs = f2/f2_sum
    d2 = np.sum(np.abs(np.cumsum(f1_probs) - np.cumsum(f2_probs)))


    return sigma*d1 + d2
    
    

def _distance_function(u, v, is_spectral, nchannels, syllable_path_pairs=None, how_many_bins=-1, align=True, return_aligned_arrays=False):
    '''
    u and v are vectors from the spectral or neural data, with the 0th entry representing rendition index.
    The 0th entry is to be used for dtw alignment.
    This function returns the aligned and averaged(along channels) distance.
    '''
    if align and syllable_path_pairs is None:
        raise ValueError('To align (time warp), syllable_path_pairs should not be None.')
    
    ### u and v are vectorized form of (nchannels, number of bins)
    if how_many_bins==-1:
        if len(u.shape)==1:
            how_many_bins = (len(u)-1)/nchannels
        else:
            how_many_bins = (u.shape[1]-1)/nchannels
            
    
    ## extract id for pair
    u_id = u[0]
    v_id = v[0]
    try:
        u_values_2d = u[1:].reshape((nchannels,-1))
        v_values_2d = v[1:].reshape((nchannels,-1))
    except:
        u_values_2d = u[0:].reshape((nchannels,-1))
        v_values_2d = v[0:].reshape((nchannels,-1))


    ### define which metric to use
    if is_spectral:
        metric = distance.euclidean
        pair_step_size = 1
    else:
        metric = lambda u,v : emd_extension(u, v)
        if align: ## needed only if you wish to align
            num_segs = max(syllable_path_pairs[(u_id, v_id)][-1])+1  ## number of segments = last pair index, meaning the id of the last spectral segment 
            pair_step_size = how_many_bins/num_segs  ## how many neural_segments correspond to one spectral segment
    
    u_aligned_2d = []
    v_aligned_2d = []
    
    if align:
        for channel in range(nchannels):
            '''
            The following is a labyrinth of a code to match the neural data with the spectral time warped indices.
            pair_step_size : how many neural segments correspond to one spectral segment (ex. if 6neural & 15spectral, 0.4 neural segments correspond to 1 spectral segment)
    
            If pair_step_size<1, i.e. less than one neural segments correspond to one spectral segment, we duplicate the neural segments to match a spectral segment.
            If more than one neural segments correspond to one spectral segment, we take multiple neural segments to match a spectral segment.
            '''
            if pair_step_size < 1:  
                u_aligned_strip = []
                v_aligned_strip = []
                for pair in syllable_path_pairs[(u_id, v_id)]:
                    u_col_idx = int(pair[0]*pair_step_size)
                    v_col_idx = int(pair[1]*pair_step_size)
                    
                    u_aligned_strip.append(u_values_2d[channel, u_col_idx])
                    v_aligned_strip.append(v_values_2d[channel, v_col_idx])
        
                u_aligned_strip = np.array(u_aligned_strip)   ## vector of length (num warped pairs), num warped pairs >= number of segments because time warping results in duplicate accessing of segments
                v_aligned_strip = np.array(v_aligned_strip)
                
            else:
                u_aligned_strip = np.array([u_values_2d[channel,int(pair[0]*pair_step_size):int((pair[0]+1)*pair_step_size)] for pair in syllable_path_pairs[(u_id, v_id)]])  ##shape (num warped pairs, number of segments)
                v_aligned_strip = np.array([v_values_2d[channel,int(pair[1]*pair_step_size):int((pair[1]+1)*pair_step_size)] for pair in syllable_path_pairs[(u_id, v_id)]])
        
            u_aligned_2d.append(u_aligned_strip.flatten())
            v_aligned_2d.append(v_aligned_strip.flatten())
    else:
        for channel in range(nchannels):
            u_aligned_strip = np.array(u_values_2d[channel,:])
            v_aligned_strip = np.array(v_values_2d[channel,:])
            
            u_aligned_2d.append(u_aligned_strip)
            v_aligned_2d.append(v_aligned_strip)
            
    u_aligned_2d = np.array(u_aligned_2d)  ## shape (nchannels, num warped pairs * number of segments)
    v_aligned_2d = np.array(v_aligned_2d)  ## shape (nchannels, num warped pairs * number of segments)
    
    
    pairwise_distances = []
    for i in range(nchannels):
        dist = metric(u_aligned_2d[i,:], v_aligned_2d[i,:])
        pairwise_distances.append(dist)
    
    if return_aligned_arrays:
        return u_aligned_2d, v_aligned_2d
    else:
        return np.mean(pairwise_distances)
    
    
def visualize_aligned(sylA, sylB, is_spectral, channels_of_interest=None, syllable_path_pairs=None, neural_bin_size=None, align=True):
    '''
    Please provide sylA, sylB as vectors
    '''
    
    if is_spectral:
        sylA_aligned, sylB_aligned = _distance_function(sylA, sylB, is_spectral=is_spectral, nchannels=128, syllable_path_pairs=syllable_path_pairs, align=align, return_aligned_arrays=True)
        print(f'Syllable_A spectrogram shape {sylA_aligned.shape}, Syllable_B spectrogram shape{sylB_aligned.shape}')
        plot_spectrogram(sylA_aligned, title='Syllable_A Spectrogram')
        plot_spectrogram(sylB_aligned, title='Syllable_B Spectrogram')
        
    else:
        if channels_of_interest is None or neural_bin_size is None:
            raise ValueError('Please specify channels of interest')
        sylA_aligned, sylB_aligned = _distance_function(sylA, sylB, is_spectral=is_spectral, nchannels=len(channels_of_interest), syllable_path_pairs=syllable_path_pairs, align=align, return_aligned_arrays=True)
        print(f'Syllable_A neural shape {sylA_aligned.shape}, Syllable_B neural shape{sylB_aligned.shape}')
        plot_raster(sylA_aligned, channels_of_interest, neural_bin_size, title='Syllable_A Neural Data')
        plot_raster(sylB_aligned, channels_of_interest, neural_bin_size, title='Syllable_B Neural Data')
        
        

def continuity_computation(neural_properties_list, spec_files, audioevt_properties, channels_of_interest, neural_window, spectral_window, neural_bin_size=1, spectral_bin_size=80, n_neighbors = 10, syllable_path_pairs=None, align=True):
    '''
    Returns a pandas dataframe with the nearest neighbors and distances
    
    neural_properties_list : A list of 'neural_properties' (length of 11, one per file). One 'neural_properties' contains 'channel_id', 'file_num', and 'spikes'
    spec_files : A numpy array of spectrograms (11, one per file)
    audioevt_properties : A pandas dataframe containing 'evt_id'(rendition id), 'syllable_start'(sample id of when the syllable starts), 'duration'(in samples), and 'file_num'.
    channels_of_interest : Channels of interest. For LMAN, it's 220~300.
    neural_window : Neural window for analysis
    spectral_window : Spectral window for analysis
    neural_bin_size : Bin size (in samples) for neural data
    spectral_bin_size : Bin size (in samples) for spectral data 
    n_neighbors : Number of neighbors for KNN
    '''
    print('Computing KNNs for continuity. It may take a while...')
    
    # Neural Data
    neural_data = get_neural_activity(neural_properties_list, audioevt_properties, channels_of_interest, neural_window, neural_bin_size=neural_bin_size)
    neural_data_with_index = get_neural_activity_with_index(neural_data)

    # Spectral Data
    spectral_data = get_spectral_activity(spec_files, audioevt_properties, spectral_window, spectral_bin_size=spectral_bin_size)
    spectral_data_with_index = get_spectral_activity_with_index(spectral_data)
    
    if syllable_path_pairs is None and align:
        syllable_path_pairs = get_syllable_pairs(audioevt_properties, spectral_data)
    if not align:
        syllable_path_pairs = None
    
    
    def neural_metric(u,v):
        return _distance_function(u, v, is_spectral=False, nchannels=len(channels_of_interest), syllable_path_pairs=syllable_path_pairs, align=align)

    def spectral_metric(u,v):
        return _distance_function(u, v, is_spectral=True, nchannels=128, syllable_path_pairs=syllable_path_pairs, align=align)    

    # neural knn
    ## All num_syllables nearest neighbors are computed because we need full tables of ranked indexes and distances
    neural_kNN = kNN(n_neighbors = neural_data.shape[0], metric = neural_metric)
    neural_kNN.fit(neural_data_with_index)
    neural_distances, neural_indexes = neural_kNN.kneighbors(neural_data_with_index)  
    
    '''
    for example,
    distances[0,1] is the distance betwwen rendition#0 and rendition#i, where i is the 1st closest to rendition#0
    indexes[0,1] represents what #i is
    '''
    
    
    neural_nearest_k_ids = []
    neural_nearest_k_distances = []
    neural_nearest_k_avg_distances = []
    for id in audioevt_properties['evt_id']:
        neural_nearest_k_ids.append(neural_indexes[id][1:min(n_neighbors+1, neural_data.shape[0])])
        neural_nearest_k_distances.append(neural_distances[id][1:min(n_neighbors+1, neural_data.shape[0])])
        neural_nearest_k_avg_distances.append(np.mean(neural_distances[id][1:min(n_neighbors+1, neural_data.shape[0])]))

    audioevt_properties['NNN_id'] = neural_nearest_k_ids
    audioevt_properties['NNN_dist'] = neural_nearest_k_distances
    audioevt_properties['NNN_avg'] = neural_nearest_k_avg_distances

    # spectral knn
    ## All num_syllables nearest neighbors are computed because we need full tables of ranked indexes and distances
    spectral_kNN = kNN(n_neighbors = spectral_data.shape[0], metric = spectral_metric)
    spectral_kNN.fit(spectral_data_with_index)
    spectral_distances, spectral_indexes = spectral_kNN.kneighbors(spectral_data_with_index)

    spectral_nearest_k_ids = []
    spectral_nearest_k_distances = []
    spectral_nearest_k_avg_distances = []
    for id in audioevt_properties['evt_id']:
        spectral_nearest_k_ids.append(spectral_indexes[id][1:min(n_neighbors+1, spectral_data.shape[0])])
        spectral_nearest_k_distances.append(spectral_distances[id][1:min(n_neighbors+1, spectral_data.shape[0])])
        spectral_nearest_k_avg_distances.append(np.mean(spectral_distances[id][1:min(n_neighbors+1, spectral_data.shape[0])]))

    audioevt_properties['NSN_id'] = spectral_nearest_k_ids
    audioevt_properties['NSN_dist'] = spectral_nearest_k_distances
    audioevt_properties['NSN_avg'] = spectral_nearest_k_avg_distances

    #mapping
    ## Neural neighbors in spectral space
    neural2spectral_distances = []
    neural2spectral_avg_distances = []
    for id in audioevt_properties['evt_id']:
        neural_nearest_ids = audioevt_properties[audioevt_properties['evt_id']==id]['NNN_id'].values[0]
        spectral_idx4neural = [(np.argwhere(spectral_indexes[id] == x)).flatten()[0] for x in neural_nearest_ids]  ##spectral indexes for a neural neighbor
        corresponding_distances = spectral_distances[id, spectral_idx4neural]
        neural2spectral_distances.append(corresponding_distances)
        neural2spectral_avg_distances.append(np.mean(corresponding_distances))

    audioevt_properties['CSS_dist'] = neural2spectral_distances
    audioevt_properties['CSS_avg'] = neural2spectral_avg_distances

    ## Spectral neighbors in neural space
    spectral2neural_distances = []
    spectral2neural_avg_distances = []
    for id in audioevt_properties['evt_id']:
        spectral_nearest_ids = audioevt_properties[audioevt_properties['evt_id']==id]['NSN_id'].values[0]
        neural_idx4spectral = [(np.argwhere(neural_indexes[id] == x)).flatten()[0] for x in spectral_nearest_ids]
        corresponding_distances = neural_distances[id, neural_idx4spectral]
        spectral2neural_distances.append(corresponding_distances)
        spectral2neural_avg_distances.append(np.mean(corresponding_distances))

    audioevt_properties['CNS_dist'] = spectral2neural_distances
    audioevt_properties['CNS_avg'] = spectral2neural_avg_distances


    ## Rmin and Rrand, stable Rrand
    CSS_Rmins = []
    RSS_Ravgs = []
    CNS_Rmins = []
    RNS_Ravgs = []

    for syllable in audioevt_properties.iterrows():
        syllable = syllable[1]
        if syllable['NSN_avg']==0 or syllable['NNN_avg']==0:
            CSS_Rmins.append(np.nan)
            RSS_Ravgs.append(np.nan)
            CNS_Rmins.append(np.nan)
            RNS_Ravgs.append(np.nan)
            continue

        id = syllable['evt_id']
        CSS_Rmin = syllable['CSS_avg'] / syllable['NSN_avg']
        if min(n_neighbors+1, spectral_data.shape[0]) == spectral_distances.shape[1]:
            RSS_Ravg = np.nan
        else:
            RSS_dist = spectral_distances[id, min(n_neighbors+1, spectral_data.shape[0]):]  ## distances for the rest of the neighbors
            RSS_avg = np.mean(RSS_dist)
            RSS_Ravg = RSS_avg / syllable['NSN_avg']

        CNS_Rmin = syllable['CNS_avg'] / syllable['NNN_avg']
        if min(n_neighbors+1, neural_data.shape[0]) == neural_distances.shape[1]:
            RNS_Ravg = np.nan
        else:
            RNS_dist = neural_distances[id, min(n_neighbors+1, neural_data.shape[0]):]  ## distances for the rest of the neighbors
            RNS_avg = np.mean(RNS_dist)
            RNS_Ravg = RNS_avg / syllable['NNN_avg']

        CSS_Rmins.append(CSS_Rmin)
        RSS_Ravgs.append(RSS_Ravg)
        CNS_Rmins.append(CNS_Rmin)
        RNS_Ravgs.append(RNS_Ravg)

    audioevt_properties['CSS_Rmins'] = CSS_Rmins
    audioevt_properties['RSS_Ravgs'] = RSS_Ravgs
    audioevt_properties['CNS_Rmins'] = CNS_Rmins
    audioevt_properties['RNS_Ravgs'] = RNS_Ravgs

    return audioevt_properties
