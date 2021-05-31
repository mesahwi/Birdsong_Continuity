from misc import *
import numpy as np
import pandas as pd
import pickle
# from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as kNN


# def _distribution_convert(u):
#     '''
#     Takes in a 1D vector and outputs an empircal distribution
#     A helper function for wasserstein metric. 
#     The wasserstein metric is a distance function between two empirical distributions.
#     Therefore the metric does not take order into account, regarding the two vectors [0,1,2] and [2,1,0] as equal.
#     (ex. Think of a distribution of the heights of people in a room. Heights of [170,170,180,190]cm and heights of [190,170,180,170]cm are the same distribution)
#     With _distribution_convert(), we change the vectors into 'distribution' vectors : [1,0,2,1]=>[0,0,1,2,2,2,3,3]
#         :(meaning the 0th index appears 1(+1) times, the 1st index appears 0(+1) times, the 2nd index appears 2(+1) times, and the 3rd index appears 1(+1) times)
#         The (+1) are there so that u=[0,0,0,0] does not return an empty array
#     '''
#     d = [np.repeat(i,u[i]+1) for i in range(len(u))]
#     return np.concatenate(d)

def emd(u_vec, v_vec):
    assert len(u_vec)==len(v_vec)
    cnt = 0.
    u_cdf = np.zeros_like(u_vec, dtype=np.float)
    for i, u in enumerate(u_vec):
        cnt += u
        u_cdf[i] = cnt/len(u_vec)
        # u_cdf[i] = cnt
 
    cnt = 0.
    v_cdf = np.zeros_like(v_vec, dtype=np.float)
    for i, v in enumerate(v_vec):
        cnt += v
        v_cdf[i] = cnt/len(v_vec)
        # v_cdf[i] = cnt
 
    return np.sum(np.abs(u_cdf - v_cdf))
    
    

def _distance_with_alignment(u, v, syllable_path_pairs, is_spectral, nchannels, how_many_bins=-1):
    '''
    u and v are vectors from the spectral or neural data, with the 0th entry representing rendition index.
    The 0th entry is to be used for dtw alignment.
    This function returns the aligned and averaged(along channels) distance.
    '''
    ### u and v are vectorized form of (nchannels, number of bins)
    if how_many_bins==-1:
        if len(u.shape)==1:
            how_many_bins = (len(u)-1)/nchannels
        else:
            how_many_bins = (u.shape[1]-1)/nchannels
            
    
    ## extract id for pair
    u_id = u[0]
    v_id = v[0]

    u_values_2d = u[1:].reshape((nchannels,-1))
    v_values_2d = v[1:].reshape((nchannels,-1))


    ### define which metric to use
    if is_spectral:
        metric = distance.euclidean
        pair_step_size = 1
    else:
        metric = lambda u,v : emd(u, v)
        # metric = distance.euclidean
        num_segs = syllable_path_pairs[(u_id, v_id)][-1][0]+1  ## number of segments = last pair index, meaning the id of the last spectral segment 
        pair_step_size = how_many_bins/num_segs  ## how many neural_segments correspond to one spectral segment
    
    u_aligned_2d = []
    v_aligned_2d = []
    
    for channel in range(nchannels):
        '''
        The following is a labyrinth of a code to match the neural data with the spectral time warped indices.
        pair_step_size is how many neural segments correspond to one spectral segment
        If less than one neural segments correspond to one spectral segment, we duplicate the neural segments to match a spectral segment, then account for the duplication by multiplying the values with pair_step_size (<1)
        If more than one neural segments correspond to one spectral segment, we take multiple neural segments to match a spectral segment, but do not scale the values because no values were missing nor duplicated.
        '''
        if pair_step_size < 1:  
            u_aligned_strip = []
            v_aligned_strip = []
            for pair in syllable_path_pairs[(u_id, v_id)]:
                u_col_idx = int(pair[0]*pair_step_size)
                v_col_idx = int(pair[1]*pair_step_size)
                
                u_aligned_strip.append(u_values_2d[channel, u_col_idx])
                v_aligned_strip.append(v_values_2d[channel, v_col_idx])
    
            u_aligned_strip = np.array(u_aligned_strip)   ## shape (nchannels, num warped pairs), num warped pairs >= number of segments because time warping results in duplicate accessing of segments
            v_aligned_strip = np.array(v_aligned_strip)
            
        else:
            u_aligned_strip = np.array([u_values_2d[channel,int(pair[0]*pair_step_size):int((pair[0]+1)*pair_step_size)] for pair in syllable_path_pairs[(u_id, v_id)]])  ##shape (num warped pairs, number of segments)
            v_aligned_strip = np.array([v_values_2d[channel,int(pair[1]*pair_step_size):int((pair[1]+1)*pair_step_size)] for pair in syllable_path_pairs[(u_id, v_id)]])
    
        u_aligned_2d.append(u_aligned_strip.flatten())
        v_aligned_2d.append(v_aligned_strip.flatten())
            
    u_aligned_2d = np.array(u_aligned_2d)  ## shape (nchannels, num warped pairs * number of segments)
    v_aligned_2d = np.array(v_aligned_2d)  ## shape (nchannels, num warped pairs * number of segments)
    
    
    pairwise_distances = []
    for i in range(nchannels):
        dist = metric(u_aligned_2d[i,:], v_aligned_2d[i,:])
        pairwise_distances.append(dist)
    
    return np.mean(pairwise_distances)
    


def continuity_computation(neural_properties_list, spec_files, audioevt_properties, channels_of_interest, neural_window, spectral_window, neural_bin_size=1, spectral_bin_size=80, n_neighbors = 10):

    
    # Neural Data
    neural_data = get_neural_activity(neural_properties_list, audioevt_properties, channels_of_interest, neural_window, neural_bin_size=neural_bin_size)
    neural_data_with_index = get_neural_activity_with_index(neural_data)

    # Spectral Data
    spectral_data = get_spectral_activity(spec_files, audioevt_properties, spectral_window, spectral_bin_size=spectral_bin_size)
    spectral_data_with_index = get_spectral_activity_with_index(spectral_data)
    syllable_path_pairs = get_syllable_pairs(audioevt_properties, spectral_data)
    
    
    def neural_metric(u,v):
        return _distance_with_alignment(u, v, syllable_path_pairs, is_spectral=False, nchannels=len(channels_of_interest))

    def spectral_metric(u,v):
        return _distance_with_alignment(u, v, syllable_path_pairs, is_spectral=True, nchannels=128)    

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
        neural_nearest_k_ids.append(neural_indexes[id][1:n_neighbors+1])
        neural_nearest_k_distances.append(neural_distances[id][1:n_neighbors+1])
        neural_nearest_k_avg_distances.append(np.mean(neural_distances[id][1:n_neighbors+1]))

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
        spectral_nearest_k_ids.append(spectral_indexes[id][1:n_neighbors+1])
        spectral_nearest_k_distances.append(spectral_distances[id][1:n_neighbors+1])
        spectral_nearest_k_avg_distances.append(np.mean(spectral_distances[id][1:n_neighbors+1]))

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
        RSS_dist = spectral_distances[id, n_neighbors+1:]  ## distances for the rest of the neighbors
        RSS_avg = np.mean(RSS_dist)
        RSS_Ravg = RSS_avg / syllable['NSN_avg']

        CNS_Rmin = syllable['CNS_avg'] / syllable['NNN_avg']
        RNS_dist = neural_distances[id, n_neighbors+1:]  ## distances for the rest of the neighbors
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