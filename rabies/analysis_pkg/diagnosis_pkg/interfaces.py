import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from rabies.analysis_pkg.diagnosis_pkg import diagnosis_functions

from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, BaseInterface
)

class PrepMasksInputSpec(BaseInterfaceInputSpec):
    mask_dict_list = traits.List(
        exists=True, mandatory=True, desc="Brain mask.")
    prior_maps = File(exists=True, mandatory=True,
                      desc="MELODIC ICA components to use.")
    DSURQE_regions = traits.Bool(
        desc="Whether to use the regional masks generated from the DSURQE atlas for the grayplots outputs. Requires using the DSURQE template for preprocessing.")


class PrepMasksOutputSpec(TraitedSpec):
    mask_file_dict = traits.Dict(
        desc="A dictionary regrouping the all required accompanying files.")


class PrepMasks(BaseInterface):
    """

    """

    input_spec = PrepMasksInputSpec
    output_spec = PrepMasksOutputSpec

    def _run_interface(self, runtime):
        from rabies.utils import flatten_list,resample_image_spacing
        merged = flatten_list(list(self.inputs.mask_dict_list))
        mask_dict = merged[0]  # all mask files are assumed to be identical
        brain_mask_file = mask_dict['mask_file']
        WM_mask_file = mask_dict['WM_mask_file']
        CSF_mask_file = mask_dict['CSF_mask_file']

        # resample the template to the EPI dimensions
        resampled = resample_image_spacing(sitk.ReadImage(mask_dict['preprocess_anat_template']), sitk.ReadImage(
            brain_mask_file).GetSpacing())
        template_file = os.path.abspath('display_template.nii.gz')
        sitk.WriteImage(resampled, template_file)

        if self.inputs.DSURQE_regions:
            if 'XDG_DATA_HOME' in os.environ.keys():
                rabies_path = os.environ['XDG_DATA_HOME']+'/rabies'
            else:
                rabies_path = os.environ['HOME']+'/.local/share/rabies'
            right_hem_mask_file = diagnosis_functions.resample_mask(rabies_path+'/DSURQE_40micron_right_hem_mask.nii.gz',
                                                brain_mask_file)
            left_hem_mask_file = diagnosis_functions.resample_mask(rabies_path+'/DSURQE_40micron_left_hem_mask.nii.gz',
                                               brain_mask_file)
        else:
            right_hem_mask_file = ''
            left_hem_mask_file = ''

        from rabies.analysis_pkg.analysis_functions import resample_IC_file
        prior_maps = resample_IC_file(self.inputs.prior_maps, brain_mask_file)

        edge_mask_file = os.path.abspath('edge_mask.nii.gz')
        diagnosis_functions.compute_edge_mask(brain_mask_file, edge_mask_file, num_edge_voxels=1)
        mask_file_dict = {'template_file': template_file, 'brain_mask': brain_mask_file, 'WM_mask': WM_mask_file, 'CSF_mask': CSF_mask_file,
                          'edge_mask': edge_mask_file, 'right_hem_mask': right_hem_mask_file, 'left_hem_mask': left_hem_mask_file, 'prior_maps': prior_maps}

        setattr(self, 'mask_file_dict', mask_file_dict)
        return runtime

    def _list_outputs(self):
        return {'mask_file_dict': getattr(self, 'mask_file_dict')}


class ScanDiagnosisInputSpec(BaseInterfaceInputSpec):
    file_dict = traits.Dict(
        desc="A dictionary regrouping the all required accompanying files.")
    mask_file_dict = traits.Dict(
        desc="A dictionary regrouping the all required accompanying files.")
    analysis_dict = traits.Dict(
        desc="A dictionary regrouping relevant outputs from analysis.")
    prior_bold_idx = traits.List(
        desc="The index for the ICA components that correspond to bold sources.")
    prior_confound_idx = traits.List(
        desc="The index for the ICA components that correspond to confounding sources.")
    NPR_extra_sources = traits.Int(
        desc="number of extra components from NPR.")
    DSURQE_regions = traits.Bool(
        desc="Whether to use the regional masks generated from the DSURQE atlas for the grayplots outputs. Requires using the DSURQE template for preprocessing.")


class ScanDiagnosisOutputSpec(TraitedSpec):
    figure_temporal_diagnosis = File(
        exists=True, desc="Output figure from the scan diagnosis")
    figure_spatial_diagnosis = File(
        exists=True, desc="Output figure from the scan diagnosis")
    temporal_info = traits.Dict(
        desc="A dictionary regrouping the temporal features.")
    spatial_info = traits.Dict(
        desc="A dictionary regrouping the spatial features.")


class ScanDiagnosis(BaseInterface):
    """
    Extracts several spatial and temporal features on the target scan.
    Spatial features include tSTD, CR-R^2 (variance explained from confound regression),
    correlation maps with global signal/DVARS/FD, and network maps from specified
    BOLD priors at the indices of prior_bold_idx.
    Temporal features include grayplot, 6 motion parameters, framewise displacement,
    DVARS, WM/CSV/edge mask timecourses, CR-R^2, and the average amplitude of BOLD and
    confound components seperately.
    """

    input_spec = ScanDiagnosisInputSpec
    output_spec = ScanDiagnosisOutputSpec

    def _run_interface(self, runtime):
        # convert to an integer list
        bold_file = self.inputs.file_dict['bold_file']
        CR_data_dict = self.inputs.file_dict['CR_data_dict']
        VE_file = self.inputs.file_dict['VE_file']
        STD_file = self.inputs.file_dict['STD_file']
        CR_STD_file = self.inputs.file_dict['CR_STD_file']
        random_CR_STD_file = self.inputs.file_dict['random_CR_STD_file']
        corrected_CR_STD_file = self.inputs.file_dict['corrected_CR_STD_file']
        prior_bold_idx = [int(i) for i in self.inputs.prior_bold_idx]
        prior_confound_idx = [int(i) for i in self.inputs.prior_confound_idx]

        temporal_info, spatial_info = diagnosis_functions.process_data(
            bold_file, CR_data_dict, VE_file, STD_file, CR_STD_file, random_CR_STD_file, corrected_CR_STD_file, self.inputs.mask_file_dict, self.inputs.analysis_dict, prior_bold_idx, prior_confound_idx, NPR_extra_sources=self.inputs.NPR_extra_sources)

        fig, fig2 = diagnosis_functions.scan_diagnosis(bold_file, self.inputs.mask_file_dict, temporal_info,
                                   spatial_info, CR_data_dict, regional_grayplot=self.inputs.DSURQE_regions)

        import pathlib
        filename_template = pathlib.Path(bold_file).name.rsplit(".nii")[0]
        figure_path = os.path.abspath(filename_template)
        fig.savefig(figure_path+'_temporal_diagnosis.png', bbox_inches='tight')
        fig2.savefig(figure_path+'_spatial_diagnosis.png', bbox_inches='tight')

        setattr(self, 'figure_temporal_diagnosis',
                figure_path+'_temporal_diagnosis.png')
        setattr(self, 'figure_spatial_diagnosis',
                figure_path+'_spatial_diagnosis.png')
        setattr(self, 'temporal_info', temporal_info)
        setattr(self, 'spatial_info', spatial_info)

        return runtime

    def _list_outputs(self):
        return {'figure_temporal_diagnosis': getattr(self, 'figure_temporal_diagnosis'),
                'figure_spatial_diagnosis': getattr(self, 'figure_spatial_diagnosis'),
                'temporal_info': getattr(self, 'temporal_info'),
                'spatial_info': getattr(self, 'spatial_info'), }


class DatasetDiagnosisInputSpec(BaseInterfaceInputSpec):
    scan_data_list = traits.List(
        exists=True, mandatory=True, desc="A dictionary regrouping the all required accompanying data per scan.")
    mask_file_dict = traits.Dict(
        exists=True, mandatory=True, desc="A dictionary regrouping the all required accompanying files.")
    seed_prior_maps = traits.List(
        exists=True, desc="A list of expected network map associated to each seed-FC.")


class DatasetDiagnosisOutputSpec(TraitedSpec):
    analysis_QC = traits.Str(
        exists=True, desc="Output figure from the analysis QC.")


class DatasetDiagnosis(BaseInterface):
    """
    Conducts a group-level correlation analysis to assess artefact effects.
    Computes the voxelwise cross-subject correlation between each spatial features
    from the previously run scan diagnosis.
    """

    input_spec = DatasetDiagnosisInputSpec
    output_spec = DatasetDiagnosisOutputSpec

    def _run_interface(self, runtime):
        from rabies.utils import flatten_list
        from .analysis_QC import analysis_QC

        merged = flatten_list(list(self.inputs.scan_data_list))
        if len(merged) < 3:
            raise ValueError(
                "Cannot run statistics on a sample size smaller than 3, so an empty figure is generated.")

        out_dir_normal = os.path.abspath('analysis_QC')
        os.makedirs(out_dir_normal, exist_ok=True)
        out_dir_corrected = f'{out_dir_normal}/analysis_QC_corrected_CR'
        os.makedirs(out_dir_corrected, exist_ok=True)

        template_file = self.inputs.mask_file_dict['template_file']
        mask_file = self.inputs.mask_file_dict['brain_mask']
        brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        volume_indices = brain_mask.astype(bool)

        std_maps=[]
        CR_std_maps=[]
        random_CR_std_maps=[]
        corrected_CR_std_maps=[]
        DR_maps_list=[]
        seed_maps_list=[]
        NPR_maps_list=[]
        tdof_list=[]
        for scan_data in merged:
            std_maps.append(scan_data['temporal_std'])
            CR_std_maps.append(scan_data['predicted_std'])
            corrected_CR_std_maps.append(scan_data['corrected_CR_std'])
            random_CR_std_maps.append(scan_data['random_CR_std'])
            DR_maps_list.append(scan_data['DR_BOLD'])
            NPR_maps_list.append(scan_data['NPR_maps'])
            tdof_list.append(scan_data['tDOF'])
            seed_maps_list.append(scan_data['seed_list'])

        from rabies.utils import recover_3D
        non_zero_voxels = ((std_maps==0).sum(axis=0).astype(bool)==0)
        non_zero_mask = os.path.abspath('non_zero_mask.nii.gz')
        sitk.WriteImage(recover_3D(mask_file, non_zero_voxels.astype(float)), non_zero_mask)

        std_maps=np.array(std_maps)[:,non_zero_voxels]
        CR_std_maps=np.array(CR_std_maps)[:,non_zero_voxels]
        corrected_CR_std_maps=np.array(corrected_CR_std_maps)[:,non_zero_voxels]
        random_CR_std_maps=np.array(random_CR_std_maps)[:,non_zero_voxels]

        for corr_variable,variable_name,out_dir in zip(
            [[std_maps, CR_std_maps], [std_maps, corrected_CR_std_maps, random_CR_std_maps]], 
            [['$\mathregular{BOLD_{SD}}$', '$\mathregular{CR_{SD}}$'], ['$\mathregular{BOLD_{SD}}$', 'Corrected $\mathregular{CR_{SD}}$', 'Random $\mathregular{CR_{SD}}$']], 
            [out_dir_normal, out_dir_corrected]):

            # tdof effect; if there's no variability don't compute
            if not np.array(tdof_list).std()==0:
                tdof = np.array(tdof_list).reshape(-1,1)
                corr_variable.append(tdof)
                variable_name.append('tDOF')

            prior_maps = scan_data['prior_maps'][:,non_zero_voxels]
            num_priors = prior_maps.shape[0]

            DR_maps_list=np.array(DR_maps_list)
            for i in range(num_priors):
                FC_maps = DR_maps_list[:,i,non_zero_voxels]
                fig_path = f'{out_dir}/DR{i}_QC_maps.png'
                dataset_stats = analysis_QC(FC_maps, prior_maps[i,:], non_zero_mask, corr_variable, variable_name, template_file, fig_path)
                pd.DataFrame(dataset_stats, index=[1]).to_csv(f'{out_dir}/DR{i}_QC_stats.csv', index=None)

            NPR_maps_list=np.array(NPR_maps_list)
            if NPR_maps_list.shape[1]>0:
                for i in range(num_priors):
                    FC_maps = NPR_maps_list[:,i,non_zero_voxels]
                    fig_path = f'{out_dir}/NPR{i}_QC_maps.png'
                    dataset_stats = analysis_QC(FC_maps, prior_maps[i,:], non_zero_mask, corr_variable, variable_name, template_file, fig_path)
                    pd.DataFrame(dataset_stats, index=[1]).to_csv(f'{out_dir}/NPR{i}_QC_stats.csv', index=None)


            # prior maps are provided for seed-FC, tries to run the diagnosis on seeds
            if len(self.inputs.seed_prior_maps)>0:
                prior_maps=[]
                for prior_map in self.inputs.seed_prior_maps:
                    # resample to match the subject
                    sitk_img = sitk.Resample(sitk.ReadImage(prior_map), sitk.ReadImage(mask_file))
                    prior_maps.append(sitk.GetArrayFromImage(sitk_img)[volume_indices])

                prior_maps = np.array(prior_maps)[:,non_zero_voxels]
                num_priors = prior_maps.shape[0]
                seed_maps_list=np.array(seed_maps_list)
                for i in range(num_priors):
                    FC_maps = seed_maps_list[:,i,non_zero_voxels]
                    fig_path = f'{out_dir}/seed_FC{i}_QC_maps.png'
                    dataset_stats = analysis_QC(FC_maps, prior_maps[i,:], non_zero_mask, corr_variable, variable_name, template_file, fig_path)
                    pd.DataFrame(dataset_stats, index=[1]).to_csv(f'{out_dir}/seed_FC{i}_QC_stats.csv', index=None)

        setattr(self, 'analysis_QC',
                out_dir_normal)
        return runtime

    def _list_outputs(self):
        return {
            'analysis_QC': getattr(self, 'analysis_QC'),
            }

