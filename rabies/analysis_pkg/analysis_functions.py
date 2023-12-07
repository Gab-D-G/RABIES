import numpy as np
import SimpleITK as sitk
from .analysis_math import vcorrcoef,closed_form


'''
seed-based FC
'''

def seed_based_FC(dict_file, seed_dict, seed_name):
    import os
    import numpy as np
    import SimpleITK as sitk
    import pathlib
    from rabies.utils import run_command, recover_3D
    from rabies.analysis_pkg.analysis_math import vcorrcoef

    import pickle
    with open(dict_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    bold_file = data_dict['bold_file']
    mask_file = data_dict['mask_file']
    timeseries = data_dict['timeseries']
    volume_indices = data_dict['volume_indices']

    seed_file = seed_dict[seed_name]
    resampled = os.path.abspath('resampled.nii.gz')
    command=f'antsApplyTransforms -i {seed_file} -r {mask_file} -o {resampled} -n GenericLabel'
    rc,c_out = run_command(command)
    roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(resampled))[volume_indices].astype(bool)

    # extract the voxel timeseries within the mask, and take the mean ROI timeseries
    seed_timeseries = timeseries[:,roi_mask].mean(axis=1)

    corrs = vcorrcoef(timeseries.T, seed_timeseries)
    corrs[np.isnan(corrs)] = 0

    corr_map_img = recover_3D(mask_file, corrs)
    filename_split = pathlib.Path(bold_file).name.rsplit(".nii")[0]
    corr_map_file = os.path.abspath(
        filename_split+'_'+seed_name+'_corr_map.nii.gz')
    
    sitk.WriteImage(corr_map_img, corr_map_file)

    # also save the seed timecourse
    import pandas as pd
    seed_timecourse_csv = os.path.abspath(
        filename_split+'_'+seed_name+'_timecourse.csv')
    pd.DataFrame(seed_timeseries).to_csv(seed_timecourse_csv, header=False, index=False)

    return corr_map_file,seed_timecourse_csv


'''
FC matrix
'''


def run_FC_matrix(dict_file, figure_format, roi_type='parcellated'):
    import os
    import pandas as pd
    import SimpleITK as sitk
    import numpy as np
    import pathlib  # Better path manipulation
    from rabies.analysis_pkg.analysis_functions import parcellated_FC_matrix, plot_matrix

    import pickle
    with open(dict_file, 'rb') as handle:
        data_dict = pickle.load(handle)


    bold_file = data_dict['bold_file']
    filename_split = pathlib.Path(bold_file).name.rsplit(".nii")
    figname = os.path.abspath(filename_split[0]+f'_FC_matrix.{figure_format}')
    
    timeseries = data_dict['timeseries']
    atlas_idx = data_dict['atlas_idx']
    roi_list = data_dict['roi_list']

    if roi_type == 'parcellated':
        corr_matrix,roi_labels = parcellated_FC_matrix(timeseries, atlas_idx, roi_list)
        matrix_df = pd.DataFrame(corr_matrix, index=roi_labels, columns=roi_labels)
    elif roi_type == 'voxelwise':
        corr_matrix = np.corrcoef(timeseries.T)
        matrix_df = pd.DataFrame(corr_matrix)
    else:
        raise ValueError(
            f"Invalid --ROI_type provided: {roi_type}. Must be either 'parcellated' or 'voxelwise.'")
    plot_matrix(figname, corr_matrix)

    data_file = os.path.abspath(filename_split[0]+'_FC_matrix.csv')
    matrix_df.to_csv(data_file, sep=',')
    return data_file, figname


def parcellated_FC_matrix(sub_timeseries, atlas_idx, roi_list):
    
    timeseries_dict = {}
    for i in roi_list:
        roi_mask = np.asarray(atlas_idx == i, dtype=bool)
        # extract the voxel timeseries within the mask, and take the mean ROI timeseries
        timeseries_dict[str(i)] = sub_timeseries[:,roi_mask].mean(axis=1)

    roi_labels = list(timeseries_dict.keys())
    sub_timeseries = []
    for roi in roi_labels:
        sub_timeseries.append(timeseries_dict[roi])
    corr_matrix = np.corrcoef(sub_timeseries)
    return corr_matrix,roi_labels


def plot_matrix(filename, corr_matrix):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    g = ax.imshow(corr_matrix, cmap='coolwarm', vmax=1, vmin=-1)
    ax.axis('off')
    cbar = plt.colorbar(g, ax=ax, shrink=0.5)
    cbar.set_label('R score', rotation=270, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)


'''
ICA
'''


def run_group_ICA(bold_file_list, mask_file, dim, random_seed):
    import os
    import pandas as pd

    # create a filelist.txt
    file_path = os.path.abspath('filelist.txt')
    from rabies.utils import flatten_list
    merged = flatten_list(list(bold_file_list))
    df = pd.DataFrame(data=merged)
    df.to_csv(file_path, header=False, sep=',', index=False)

    from rabies.utils import run_command
    out_dir = os.path.abspath('group_melodic.ica')
    command = f'melodic -i {file_path} -m {mask_file} -o {out_dir} -d {dim} --report --seed={str(random_seed)}'
    rc,c_out = run_command(command)
    IC_file = out_dir+'/melodic_IC.nii.gz'
    return out_dir, IC_file


def run_DR_ICA(dict_file,network_weighting):
    import os
    import pandas as pd
    import pathlib  # Better path manipulation
    import SimpleITK as sitk
    from rabies.utils import recover_4D
    from rabies.analysis_pkg.analysis_math import dual_regression

    import pickle
    with open(dict_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    bold_file = data_dict['bold_file']
    mask_file = data_dict['mask_file']
    timeseries = data_dict['timeseries']
    prior_map_vectors = data_dict['prior_map_vectors']

    filename_split = pathlib.Path(bold_file).name.rsplit(".nii")

    DR = dual_regression(prior_map_vectors, timeseries)

    if network_weighting=='absolute':
        DR_C = DR['C']*DR['S']
    elif network_weighting=='relative':
        DR_C = DR['C']
    else:
        raise 

    dual_regression_timecourse_csv = os.path.abspath(filename_split[0]+'_dual_regression_timecourse.csv')
    pd.DataFrame(DR['W']).to_csv(dual_regression_timecourse_csv, header=False, index=False)

    # save the subjects' IC maps as .nii file
    DR_maps_filename = os.path.abspath(filename_split[0]+'_DR_maps.nii.gz')
    sitk.WriteImage(recover_4D(mask_file, DR_C.T, bold_file), DR_maps_filename)
    return DR_maps_filename, dual_regression_timecourse_csv


from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, BaseInterface
)

class NeuralPriorRecoveryInputSpec(BaseInterfaceInputSpec):
    dict_file = File(exists=True, mandatory=True, desc="Dictionary with prepared analysis data.")
    prior_bold_idx = traits.List(desc="The index for the ICA components that correspond to bold sources.")
    NPR_temporal_comp = traits.Int(
        desc="number of data-driven temporal components to compute.")
    NPR_spatial_comp = traits.Int(
        desc="number of data-driven spatial components to compute.")
    optimize_NPR_dict = traits.Dict(
        desc="Dictionary with options for optimizing NPR convergence.")
    network_weighting = traits.Str(
        desc="Whether to derive absolute or relative (variance-normalized) network maps.")
    figure_format = traits.Str(
        desc="Select file format for figures.")
    CPCA_DR = traits.Bool(
        desc="Whether to use dual regression to replace NPR stage 2.")

class NeuralPriorRecoveryOutputSpec(TraitedSpec):
    NPR_prior_timecourse_csv = File(
        exists=True, desc=".csv with timecourses from the fitted prior sources.")
    NPR_extra_timecourse_csv = File(
        exists=True, desc=".csv with timecourses from the scan-specific extra sources.")
    NPR_prior_filename = File(
        exists=True, desc=".nii file with spatial components from the fitted prior sources.")
    NPR_extra_filename = File(
        exists=True, desc=".nii file with spatial components from the scan-specific extra sources.")
    optimize_report = traits.Any(
        exists=False, desc="The NPR optimization report.")


class NeuralPriorRecovery(BaseInterface):
    """

    """

    input_spec = NeuralPriorRecoveryInputSpec
    output_spec = NeuralPriorRecoveryOutputSpec

    def _run_interface(self, runtime):
        import os
        import numpy as np
        import pandas as pd
        import pathlib  # Better path manipulation
        from rabies.utils import recover_4D
        from rabies.analysis_pkg.analysis_math import spatiotemporal_prior_fit,spatiotemporal_C_PCA_DR

        import pickle
        with open(self.inputs.dict_file, 'rb') as handle:
            data_dict = pickle.load(handle)
        bold_file = data_dict['bold_file']
        mask_file = data_dict['mask_file']
        timeseries = data_dict['timeseries']
        prior_map_vectors = data_dict['prior_map_vectors']
        network_weighting=self.inputs.network_weighting

        filename_split = pathlib.Path(
            bold_file).name.rsplit(".nii")

        C_prior = prior_map_vectors[self.inputs.prior_bold_idx,:].T

        optimize_NPR_dict = self.inputs.optimize_NPR_dict
        if optimize_NPR_dict['apply']:
            modeling,_,_,_,optimize_report_fig = spatiotemporal_fit_converge(timeseries,C_prior,
                                        window_size=optimize_NPR_dict['window_size'],
                                        min_prior_corr=optimize_NPR_dict['min_prior_corr'],
                                        diff_thresh=optimize_NPR_dict['diff_thresh'],
                                        max_iter=optimize_NPR_dict['max_iter'], 
                                        compute_max=optimize_NPR_dict['compute_max'], 
                                        gen_report=True, CPCA_DR=self.inputs.CPCA_DR)
            optimize_report_file = os.path.abspath(f'{filename_split[0]}_NPR_optimize.{self.inputs.figure_format}')
            optimize_report_fig.savefig(optimize_report_file, bbox_inches='tight')
        else:
            NPR_temporal_comp = self.inputs.NPR_temporal_comp
            NPR_spatial_comp = self.inputs.NPR_spatial_comp
            if NPR_temporal_comp<1: # make sure there is no negative number
                NPR_temporal_comp=0
            if NPR_spatial_comp<1: # make sure there is no negative number
                NPR_spatial_comp=0
            if self.inputs.CPCA_DR:
                modeling = spatiotemporal_C_PCA_DR(timeseries, C_prior, num_W=NPR_temporal_comp, num_C=NPR_spatial_comp)
            else:
                modeling = spatiotemporal_prior_fit(timeseries, C_prior, num_W=NPR_temporal_comp, num_C=NPR_spatial_comp)
            optimize_report_file=None

        # put together the spatial and temporal extra components
        W_extra = np.concatenate((modeling['W_spatial'],modeling['W_temporal']), axis=1)

        if network_weighting=='absolute':
            C_extra = np.concatenate((modeling['C_spatial']*modeling['S_spatial'],
                modeling['C_temporal']*modeling['S_temporal']), axis=1)
            C_fit = modeling['C_fitted_prior']*modeling['S_fitted_prior']
        elif network_weighting=='relative':
            C_extra = np.concatenate((modeling['C_spatial'],
                modeling['C_temporal']), axis=1)
            C_fit = modeling['C_fitted_prior']
        else:
            raise 

        NPR_prior_timecourse_csv = os.path.abspath(filename_split[0]+'_NPR_prior_timecourse.csv')
        pd.DataFrame(modeling['W_fitted_prior']).to_csv(NPR_prior_timecourse_csv, header=False, index=False)

        NPR_extra_timecourse_csv = os.path.abspath(filename_split[0]+'_NPR_extra_timecourse.csv')
        pd.DataFrame(W_extra).to_csv(NPR_extra_timecourse_csv, header=False, index=False)

        NPR_prior_filename = os.path.abspath(filename_split[0]+'_NPR_prior.nii.gz')
        sitk.WriteImage(recover_4D(mask_file,C_fit.T, bold_file), NPR_prior_filename)

        if (self.inputs.NPR_temporal_comp+self.inputs.NPR_spatial_comp)>0:
            NPR_extra_filename = os.path.abspath(filename_split[0]+'_NPR_extra.nii.gz')
            sitk.WriteImage(recover_4D(mask_file,C_extra.T, bold_file), NPR_extra_filename)
        else:
            empty_img = sitk.GetImageFromArray(np.empty([1,1]))
            empty_file = os.path.abspath('empty.nii.gz')
            sitk.WriteImage(empty_img, empty_file)
            NPR_extra_filename = empty_file

        setattr(self, 'NPR_prior_timecourse_csv', NPR_prior_timecourse_csv)
        setattr(self, 'NPR_extra_timecourse_csv', NPR_extra_timecourse_csv)
        setattr(self, 'NPR_prior_filename', NPR_prior_filename)
        setattr(self, 'NPR_extra_filename', NPR_extra_filename)
        setattr(self, 'optimize_report', optimize_report_file)

        return runtime

    def _list_outputs(self):
        return {'NPR_prior_timecourse_csv': getattr(self, 'NPR_prior_timecourse_csv'),
                'NPR_extra_timecourse_csv': getattr(self, 'NPR_extra_timecourse_csv'),
                'NPR_prior_filename': getattr(self, 'NPR_prior_filename'),
                'NPR_extra_filename': getattr(self, 'NPR_extra_filename'),
                'optimize_report': getattr(self, 'optimize_report'),
                }


def eval_convergence(prior_corr_list,fit_diff_list,window_size=5,min_prior_corr=0.5,diff_thresh=0.04):
    if len(fit_diff_list)<window_size:
        return None
    for i in range(1,len(fit_diff_list)-window_size+1): # we start at 1 because idx 0 is nan
        if (np.array(prior_corr_list[i-1])<min_prior_corr).max():
            continue # if the preceding iteration does not pass corr threshold, search for another window
        window = np.array(fit_diff_list[i:i+window_size])
        if not (window>diff_thresh).max(): # if all diff are below threshold, return the first index
            return i-1 # we take -1 because the iteration preceding the window is ideal
    return None


def spatiotemporal_fit_converge(X,C_prior,window_size=5,min_prior_corr=0.5,diff_thresh=0.03,max_iter=20, compute_max=False, gen_report=False, CPCA_DR=False):
    '''
    NPR is conducted while incrementally increasing the number of components (alternating 
    between a spatial and a temporal component). 
    
    Convergence criterion 1: Iterations continue until the correlation between the fitted
    component and the prior does not reach the specified minimum.
    
    Convergence Criterion 2: At each iteration, the difference between the previous and new 
    output is evaluated (0=perfectly correlated; 1=uncorrelated). The forming set of 
    successive iterations (within a certain window length) is evaluated, and when a set 
    respects the convergence threshold for each iteration within the window, the iteration 
    preceding that window is selected as optimal output. We take the iteration preceding 
    the  window, as this corresponds to the last iteration which generated changes above 
    threshold. The sliding-window approach is employed to prevent falling within a local 
    minima, when further ameliorations may be possible with further iterations.
    
    When multiple priors are fitted, they are all simultaneously subjected to the evaluation
    of convergence, and as long as one prior fit does not meet the thresholds, iterations
    continue.
    
    X: timeseries of shape time by voxel
    C_prior: set of priors of shape voxel by num priors
    window_size: the size of the sliding window
    min_prior_corr: the minimum correlation with the prior
    diff_thresh: the convergence threshold
    max_iter: maximum number of components to derive
    '''
    
    from rabies.analysis_pkg.analysis_math import spatiotemporal_prior_fit,spatiotemporal_C_PCA_DR

    num_prior = C_prior.shape[1]
    num_list=range(0,max_iter)

    fit_list=[]
    # first entry is NaN, since there is no comparison to make
    fit_diff_list=[np.repeat(np.nan,num_prior)] 
    NPR_dict_list=[]
    prior_corr_list=[]
    for num in num_list:
        num_C = int(num/2)
        if num%2:
            num_W = num_C+1
        else:
            num_W = num_C

        if CPCA_DR:
            NPR_dict = spatiotemporal_C_PCA_DR(X, C_prior, num_W=num_W, num_C=num_C)
        else:
            NPR_dict = spatiotemporal_prior_fit(X, C_prior, num_W=num_W, num_C=num_C)
        NPR_dict_list.append(NPR_dict)
        prior_corr_list.append(NPR_dict['corr_list'])
        C_fit = NPR_dict['C_fitted_prior']
        C_fit /= np.sqrt((C_fit ** 2).sum(axis=0))
        
        if len(fit_list)>0:
            ##### evaluate convergence
            C = C_fit
            C_prev = fit_list[-1]
            # each prior has a separate lim evaluation
            lim = np.abs(np.abs((C * C_prev).sum(axis=0)) - 1)
            fit_diff_list.append(lim)

        convergence_idx = eval_convergence(prior_corr_list,fit_diff_list,window_size=window_size,min_prior_corr=min_prior_corr,diff_thresh=diff_thresh)
        if not compute_max and not convergence_idx is None:
            if gen_report:
                fig = generate_convergence_report(convergence_idx,fit_diff_list,prior_corr_list,min_prior_corr=min_prior_corr,diff_thresh=diff_thresh)
            return NPR_dict_list[convergence_idx],convergence_idx,fit_diff_list,prior_corr_list,fig
        
        fit_list.append(C_fit)

    if gen_report:
        fig = generate_convergence_report(convergence_idx,fit_diff_list,prior_corr_list,min_prior_corr=min_prior_corr,diff_thresh=diff_thresh)
    if convergence_idx is None: # if there was no convergence, the last iteration is outputed
        convergence_idx = -1
    return NPR_dict_list[convergence_idx],None,fit_diff_list,prior_corr_list,fig


def generate_convergence_report(convergence_idx,fit_diff_list,prior_corr_list,min_prior_corr=0.5,diff_thresh=0.03):
    import matplotlib.pyplot as plt

    num_list=range(len(fit_diff_list))
    num_priors = len(prior_corr_list[0])
    
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    
    axes[0].plot(num_list,prior_corr_list)
    axes[1].plot(num_list,fit_diff_list)
    if not convergence_idx is None:
        axes[0].scatter([num_list[convergence_idx]]*num_priors,prior_corr_list[convergence_idx], color='r', marker='*', s=80)
        axes[1].scatter([num_list[convergence_idx]]*num_priors,fit_diff_list[convergence_idx], color='r', marker='*', s=80)
        
    axes[0].set_ylim([0,1])
    axes[0].set_ylabel('Correlation with prior', color='white', fontsize=15)
    axes[1].set_ylim([0,min(diff_thresh*4,1)])
    axes[1].set_ylabel('Difference from previous', color='white', fontsize=15)
    
    axes[0].set_xlabel('Number of non-prior components', color='white', fontsize=15)
    axes[0].set_xlim([-1,max(num_list)+1])
    axes[1].set_xlabel('Number of non-prior components', color='white', fontsize=15)
    axes[1].set_xlim([-1,max(num_list)+1])
    
    axes[0].plot([-1,max(num_list)+1],[min_prior_corr,min_prior_corr], color='lightgray', linestyle='--', linewidth=2)
    axes[1].plot([-1,max(num_list)+1],[diff_thresh,diff_thresh], color='lightgray', linestyle='--', linewidth=2)
    
    legend_labels = [f'Prior {i+1}' for i in range(num_priors)]
    axes[0].legend(legend_labels, fontsize=12, loc='lower right')
    axes[1].legend(legend_labels, fontsize=12, loc='upper right')
    plt.tight_layout()
    return fig