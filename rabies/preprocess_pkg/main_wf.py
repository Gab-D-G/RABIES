import os
import pathlib
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
from .inho_correction import init_inho_correction_wf
from .commonspace_reg import init_commonspace_reg_wf,inherit_unbiased_files
from .bold_main_wf import init_bold_main_wf
from .utils import BIDSDataGraber, prep_bids_iter, prep_bids_iter_pe, convert_to_RAS, correct_oblique_affine, convert_3dWarp, apply_autobox, resample_template
from . import preprocess_visual_QC

def init_main_wf(data_dir_path, output_folder, opts, name='main_wf'):
    '''
    This workflow organizes the entire processing.

    **Parameters**

        data_dir_path
            Path to the input data directory with proper BIDS folder structure.
        output_folder
            path to output folder for the workflow and datasink
        opts
            parser options for preprocess
        cr_opts
            parser options for confound_correction
        analysis_opts
            parser options for analysis

    **Outputs**


        input_bold
            Input EPIs to the preprocessing
        commonspace_resampled_template
            the anatomical commonspace template after initial resampling
        anat_preproc
            Preprocessed anatomical image after bias field correction and denoising
        anat_mask
            Brain mask inherited from the common space registration
        anat_labels
            Anatomical labels inherited from the common space registration
        WM_mask
            Eroded WM mask inherited from the common space registration
        CSF_mask
            Eroded CSF mask inherited from the common space registration
        initial_bold_ref
            Initial EPI median volume subsequently used as 3D reference EPI volume
        inho_cor_bold
            3D reference EPI volume after bias field correction
        bold_to_anat_affine
            affine transform from the EPI space to the anatomical space
        bold_to_anat_warp
            non-linear transform from the EPI space to the anatomical space
        bold_to_anat_inverse_warp
            inverse non-linear transform from the EPI space to the anatomical space
        inho_cor_bold_warped2anat
            Bias field corrected 3D EPI volume warped to the anatomical space
        native_corrected_bold
            Preprocessed EPI resampled to match the anatomical space for
            susceptibility distortion correction
        corrected_bold_ref
            3D ref EPI volume from the native EPI timeseries
        motion_params_csv
            .csv file with measured motion timecourses, used as regressors for confound
            correction: 6 rigid body motion parameters + their first temporal derivate 
            + the 12 parameters squared (24 motion parameters)
        FD_voxelwise
            Voxelwise framewise displacement (FD) measures that can be integrated
            to future confound regression.
            These measures are computed from antsMotionCorrStats.
        pos_voxelwise
            Voxel distancing across time based on rigid body movement parameters,
            which can be integrated for a voxelwise motion regression
            These measures are computed from antsMotionCorrStats.
        FD_csv
            .csv file with global framewise displacement (FD) measures
        bold_brain_mask
            EPI brain mask for native corrected bold
        bold_WM_mask
            EPI WM mask for native corrected bold
        bold_CSF_mask
            EPI CSF mask for native corrected bold
        bold_labels
            EPI anatomical labels for native corrected bold
        commonspace_bold
            Motion and SDC-corrected EPI timeseries resampled into common space
            by applying transforms from the anatomical common space registration
        commonspace_mask
            EPI brain mask for commonspace bold
        commonspace_WM_mask
            EPI WM mask for commonspace bold
        commonspace_CSF_mask
            EPI CSF mask for commonspace bold
        commonspace_vascular_mask
            EPI vascular mask for commonspace bold
        commonspace_labels
            EPI anatomical labels for commonspace bold
        std_filename
            temporal STD map of the preprocessed timeseries
        tSNR_filename
            temporal SNR map of the preprocessed timeseries
    '''

    workflow = pe.Workflow(name=name)

    # set output node
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['input_bold', 'commonspace_resampled_template', 'anat_preproc', 'initial_bold_ref', 'inho_cor_bold', 'bold_to_anat_affine',
                'bold_to_anat_warp', 'bold_to_anat_inverse_warp', 'inho_cor_bold_warped2anat', 'native_bold', 'native_bold_ref', 'motion_params_csv',
                'FD_voxelwise', 'pos_voxelwise', 'FD_csv', 'native_brain_mask', 'native_WM_mask', 'native_CSF_mask', 'native_vascular_mask', 'native_labels',
                'commonspace_bold', 'commonspace_mask', 'commonspace_WM_mask', 'commonspace_CSF_mask', 'commonspace_vascular_mask',
                'commonspace_labels', 'std_filename', 'tSNR_filename', 'raw_brain_mask']),
        name='outputnode')

    # Datasink - creates output folder for important outputs
    bold_datasink = pe.Node(DataSink(base_directory=output_folder,
                                     container="bold_datasink"),
                            name="bold_datasink")

    anat_datasink = pe.Node(DataSink(base_directory=output_folder,
                                     container="anat_datasink"),
                            name="anat_datasink")

    transforms_datasink = pe.Node(DataSink(base_directory=output_folder,
                                           container="transforms_datasink"),
                                  name="transforms_datasink")

    motion_datasink = pe.Node(DataSink(base_directory=output_folder,
                                          container="motion_datasink"),
                                 name="motion_datasink")

    import bids
    bids.config.set_option('extension_initial_dot', True)
    try:
        layout = bids.layout.BIDSLayout(data_dir_path, validate=True)
    except Exception as e:
        from nipype import logging
        log = logging.getLogger('nipype.workflow')
        log.warning(f"The BIDS compliance failed: {e} \n\nRABIES will run anyway; double-check that the right files were picked up for processing.\n")
        layout = bids.layout.BIDSLayout(data_dir_path, validate=False)

    if opts.PEDir:
        split_name, scan_info, run_iter, structural_scan_list, number_functional_scans, pe_iter = prep_bids_iter_pe(
            layout, opts.bids_filter, opts.bold_only, inclusion_list=opts.inclusion_ids, exclusion_list=opts.exclusion_ids)
    else:
        split_name, scan_info, run_iter, structural_scan_list, number_functional_scans = prep_bids_iter(
            layout, opts.bids_filter, opts.bold_only, inclusion_list=opts.inclusion_ids, exclusion_list=opts.exclusion_ids)
    '''***details on outputs from prep_bids_iter:
    split_name: a list of strings, providing a sensible name to distinguish each iterable, 
        and also necessary to link up the run iterables with a specific session later.
    scan_info: a list of dictionary including the subject ID and session # for a given 
        iterable from split_name
    pe_iter(optional): a list of dictionary, where the keys correspond to a session split from         
    split_name, and the value is a list of PEs for that split. This manages iterables
        for PEs.
    run_iter: a list of dictionary, where the keys correspond to a session split from 
        split_name, and the value is a list of runs for that split. This manages iterables
        for runs.
    structural_scan_list: the set of structural scans; used for resample_template and managing # threads
    number_functional_scans: the number of functional scans; used for managing # threads

    '''

    # setting up all iterables
    main_split = pe.Node(niu.IdentityInterface(fields=['split_name', 'scan_info']),
                         name="main_split")
    main_split.iterables = [('split_name', split_name),
                            ('scan_info', scan_info)]
    main_split.synchronize = True

    bold_selectfiles = pe.Node(BIDSDataGraber(bids_dir=data_dir_path, bids_filter=opts.bids_filter['func']),
                               name='bold_selectfiles')

    # node to conver input image to consistent RAS orientation
    bold_convert_to_RAS_node = pe.Node(Function(input_names=['img_file'],
                                                output_names=['RAS_file'],
                                                function=convert_to_RAS),
                                       name='bold_convert_to_RAS')

    if not opts.oblique2card=='none':
        if opts.oblique2card=='3dWarp':
            correct_oblique=convert_3dWarp
        elif opts.oblique2card=='affine':
            correct_oblique=correct_oblique_affine
        bold_oblique2card_node = pe.Node(Function(input_names=['input'],
                                                    output_names=['output'],
                                                    function=correct_oblique),
                                        name='bold_oblique2card')
        workflow.connect([
            (bold_selectfiles, bold_oblique2card_node, [
                ('out_file', 'input'),
                ]),
            (bold_oblique2card_node, bold_convert_to_RAS_node, [
                ('output', 'img_file'),
                ]),
            ])
    else:
        workflow.connect([
            (bold_selectfiles, bold_convert_to_RAS_node, [
                ('out_file', 'img_file'),
                ]),
            ])

    format_bold_buffer = pe.Node(niu.IdentityInterface(fields=['formatted_bold']),
                                        name="format_bold_buffer")

    if opts.bold_autobox: # apply AFNI's 3dAutobox
        bold_autobox = pe.Node(Function(input_names=['in_file'],
                                                    output_names=['out_file'],
                                                    function=apply_autobox),
                                        name='bold_autobox')
        workflow.connect([
            (bold_convert_to_RAS_node, bold_autobox, [
                ('RAS_file', 'in_file'),
                ]),
            (bold_autobox, format_bold_buffer, [
                ('out_file', 'formatted_bold'),
                ]),
            ])
    else:
        workflow.connect([
            (bold_convert_to_RAS_node, format_bold_buffer, [
                ("RAS_file", "formatted_bold"),
                ]),
            ])

    if opts.inherit_unbiased_template=='none':
        inherit_unbiased=False
        # Resample the anatomical template according to the resolution of the provided input data
        resample_template_node = pe.Node(Function(input_names=['template_file', 'mask_file', 'file_list', 'spacing', 'rabies_data_type'],
                                                output_names=[
                                                    'resampled_template', 'resampled_mask'],
                                                function=resample_template),
                                        name='resample_template', mem_gb=1*opts.scale_min_memory)
        resample_template_node.inputs.template_file = str(opts.anat_template)
        resample_template_node.inputs.mask_file = str(opts.brain_mask)
        resample_template_node.inputs.spacing = opts.anatomical_resampling
        resample_template_node.inputs.file_list = structural_scan_list
        resample_template_node.inputs.rabies_data_type = opts.data_type
    else: # inherit the atlas files from previous run
        inherit_unbiased=True
        opts, inherit_dict = inherit_unbiased_files(opts.inherit_unbiased_template, opts)
        resample_template_node = pe.Node(niu.IdentityInterface(fields=['resampled_template', 'resampled_mask']),
                                            name="resample_template")
        resample_template_node.inputs.resampled_template = inherit_dict['resampled_template']
        resample_template_node.inputs.resampled_mask = inherit_dict['resampled_mask']

    # calculate the number of scans that will be registered
    num_scan = len(structural_scan_list)
    num_procs = min(opts.local_threads, num_scan)

    EPI_target_buffer = pe.Node(niu.IdentityInterface(fields=['EPI_template', 'EPI_mask']),
                                        name="EPI_target_buffer")

    commonspace_reg_wf = init_commonspace_reg_wf(opts=opts, commonspace_masking=opts.commonspace_reg['masking'], brain_extraction=opts.commonspace_reg['brain_extraction'], keep_mask_after_extract=opts.commonspace_reg['keep_mask_after_extract'], 
                                                 template_reg=opts.commonspace_reg['template_registration'], fast_commonspace=opts.commonspace_reg['fast_commonspace'], inherit_unbiased=inherit_unbiased,
                                                 output_folder=output_folder, transforms_datasink=transforms_datasink, num_procs=num_procs, output_datasinks=True, 
                                                 joinsource_list=['main_split'], name='commonspace_reg_wf')
    if inherit_unbiased:
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.unbiased_template = inherit_dict['unbiased_template']
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.unbiased_mask = inherit_dict['unbiased_mask']
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.unbiased_to_atlas_affine = inherit_dict['unbiased_to_atlas_affine']
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.unbiased_to_atlas_warp = inherit_dict['unbiased_to_atlas_warp']
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.unbiased_to_atlas_inverse_warp = inherit_dict['unbiased_to_atlas_inverse_warp']
        commonspace_reg_wf.inputs.inherit_unbiased_inputnode.warped_unbiased = inherit_dict['warped_unbiased']

    bold_main_wf = init_bold_main_wf(opts=opts, output_folder=output_folder, number_functional_scans=number_functional_scans)

    # organizing visual QC outputs
    template_diagnosis = pe.Node(Function(input_names=['anat_template', 'opts', 'out_dir', 'figure_format'],
                                       function=preprocess_visual_QC.template_info),
                              name='template_info')
    template_diagnosis.inputs.opts = opts
    template_diagnosis.inputs.out_dir = output_folder+'/preprocess_QC_report/template_files/'
    template_diagnosis.inputs.figure_format = opts.figure_format

    bold_inho_cor_diagnosis = pe.Node(Function(input_names=['raw_img','init_denoise','warped_mask','final_denoise', 'name_source', 'out_dir', 'figure_format'],
                                       function=preprocess_visual_QC.inho_cor_diagnosis),
                              name='bold_inho_cor_diagnosis')
    bold_inho_cor_diagnosis.inputs.out_dir = output_folder+'/preprocess_QC_report/bold_inho_cor/'
    bold_inho_cor_diagnosis.inputs.figure_format = opts.figure_format

    temporal_diagnosis = pe.Node(Function(input_names=['bold_file', 'motion_params_csv', 'FD_csv', 'rabies_data_type', 'name_source', 'out_dir', 'figure_format'],
                                          output_names=[
                                            'std_filename', 'tSNR_filename'],
                                       function=preprocess_visual_QC.temporal_features),
                              name='temporal_features')
    temporal_diagnosis.inputs.out_dir = output_folder+'/preprocess_QC_report/temporal_features/'
    temporal_diagnosis.inputs.rabies_data_type = opts.data_type
    temporal_diagnosis.inputs.figure_format = opts.figure_format

    # MAIN WORKFLOW STRUCTURE #######################################################
    workflow.connect([
        (main_split, bold_selectfiles, [
            ("scan_info", "scan_info"),
            ]),
        (bold_selectfiles, outputnode, [
            ('out_file', 'input_bold'),
            ]),
        (format_bold_buffer, bold_main_wf, [
            ("formatted_bold", "inputnode.bold"),
            ]),
        (resample_template_node, template_diagnosis, [
            ("resampled_template", "anat_template"),
            ]),
        (resample_template_node, commonspace_reg_wf, [
            ("resampled_template", "template_inputnode.template_anat"),
            ("resampled_mask", "template_inputnode.template_mask"),
            ]),
        (resample_template_node, bold_main_wf, [
            ("resampled_template", "inputnode.commonspace_ref"),
            ]),
        (resample_template_node, outputnode, [
            ("resampled_template", "commonspace_resampled_template"),
            ]),
        (commonspace_reg_wf, bold_main_wf, [
            ("outputnode.native_to_commonspace_transform_list", "inputnode.native_to_commonspace_transform_list"),
            ("outputnode.native_to_commonspace_inverse_list", "inputnode.native_to_commonspace_inverse_list"),
            ("outputnode.commonspace_to_native_transform_list", "inputnode.commonspace_to_native_transform_list"),
            ("outputnode.commonspace_to_native_inverse_list", "inputnode.commonspace_to_native_inverse_list"),
            ]),
        (bold_main_wf, outputnode, [
            ("outputnode.bold_ref", "initial_bold_ref"),
            ("outputnode.corrected_EPI", "inho_cor_bold"),
            ("outputnode.native_brain_mask", "native_brain_mask"),
            ("outputnode.native_WM_mask", "native_WM_mask"),
            ("outputnode.native_CSF_mask", "native_CSF_mask"),
            ("outputnode.native_vascular_mask", "native_vascular_mask"),
            ("outputnode.native_labels", "native_labels"),
            ("outputnode.motion_params_csv", "motion_params_csv"),
            ("outputnode.FD_voxelwise", "FD_voxelwise"),
            ("outputnode.pos_voxelwise", "pos_voxelwise"),
            ("outputnode.FD_csv", "FD_csv"),
            ('outputnode.bold_to_anat_affine', 'bold_to_anat_affine'),
            ('outputnode.bold_to_anat_warp', 'bold_to_anat_warp'),
            ('outputnode.bold_to_anat_inverse_warp', 'bold_to_anat_inverse_warp'),
            ("outputnode.output_warped_bold", "inho_cor_bold_warped2anat"),
            ("outputnode.native_bold", "native_bold"),
            ("outputnode.native_bold_ref", "native_bold_ref"),
            ("outputnode.commonspace_bold", "commonspace_bold"),
            ("outputnode.commonspace_mask", "commonspace_mask"),
            ("outputnode.commonspace_WM_mask", "commonspace_WM_mask"),
            ("outputnode.commonspace_CSF_mask", "commonspace_CSF_mask"),
            ("outputnode.commonspace_vascular_mask", "commonspace_vascular_mask"),
            ("outputnode.commonspace_labels", "commonspace_labels"),
            ("outputnode.raw_brain_mask", "raw_brain_mask"),
            ]),
        (bold_main_wf, bold_inho_cor_diagnosis, [
            ("outputnode.bold_ref", "raw_img"),
            ("outputnode.init_denoise", "init_denoise"),
            ("outputnode.corrected_EPI", "final_denoise"),
            ("outputnode.denoise_mask", "warped_mask"),
            ]),
        (bold_selectfiles, bold_inho_cor_diagnosis,
         [("out_file", "name_source")]),
        (bold_main_wf, temporal_diagnosis, [
            ("outputnode.commonspace_bold", "bold_file"),
            ("outputnode.motion_params_csv", "motion_params_csv"),
            ("outputnode.FD_csv", "FD_csv"),
            ]),
        (bold_selectfiles, temporal_diagnosis,
         [("out_file", "name_source")]),
        (temporal_diagnosis, outputnode, [
            ("tSNR_filename", "tSNR_filename"),
            ("std_filename", "std_filename"),
            ]),
        ])

    if not opts.bold_only:
        run_split = pe.Node(niu.IdentityInterface(fields=['run', 'split_name']),
                            name="run_split")
        run_split.itersource = ('main_split', 'split_name')
        run_split.iterables = [('run', run_iter)]

        anat_selectfiles = pe.Node(BIDSDataGraber(bids_dir=data_dir_path, bids_filter=opts.bids_filter['anat']),
                                   name='anat_selectfiles')
        anat_selectfiles.inputs.pe = None
        anat_selectfiles.inputs.run = None

        anat_convert_to_RAS_node = pe.Node(Function(input_names=['img_file'],
                                                    output_names=['RAS_file'],
                                                    function=convert_to_RAS),
                                           name='anat_convert_to_RAS')

        if not opts.oblique2card=='none':
            if opts.oblique2card=='3dWarp':
                correct_oblique=convert_3dWarp
            elif opts.oblique2card=='affine':
                correct_oblique=correct_oblique_affine
            anat_oblique2card_node = pe.Node(Function(input_names=['input'],
                                                        output_names=['output'],
                                                        function=correct_oblique),
                                            name='anat_oblique2card')
            workflow.connect([
                (anat_selectfiles, anat_oblique2card_node, [
                    ('out_file', 'input'),
                    ]),
                (anat_oblique2card_node, anat_convert_to_RAS_node, [
                    ('output', 'img_file'),
                    ]),
                ])
        else:
            workflow.connect([
                (anat_selectfiles, anat_convert_to_RAS_node, [
                    ('out_file', 'img_file'),
                    ]),
                ])

        format_anat_buffer = pe.Node(niu.IdentityInterface(fields=['formatted_anat']),
                                            name="format_anat_buffer")

        if opts.anat_autobox: # apply AFNI's 3dAutobox
            anat_autobox = pe.Node(Function(input_names=['in_file'],
                                                        output_names=['out_file'],
                                                        function=apply_autobox),
                                            name='anat_autobox')
            workflow.connect([
                (anat_convert_to_RAS_node, anat_autobox, [
                    ('RAS_file', 'in_file'),
                    ]),
                (anat_autobox, format_anat_buffer, [
                    ('out_file', 'formatted_anat'),
                    ]),
                ])
        else:
            workflow.connect([
                (anat_convert_to_RAS_node, format_anat_buffer, [
                    ("RAS_file", "formatted_anat"),
                    ]),
                ])

        # setting anat preprocessing nodes
        anat_inho_cor_wf = init_inho_correction_wf(opts=opts, image_type='structural', output_folder=output_folder, num_procs=num_procs, name="anat_inho_cor_wf")

        workflow.connect([
            (main_split, run_split, [
                ("split_name", "split_name"),
                ]),
            (main_split, anat_selectfiles,
             [("scan_info", "scan_info")]),
            (run_split, bold_selectfiles, [
                ("run", "run"),
                ]),
            (format_anat_buffer, anat_inho_cor_wf, [
                ("formatted_anat", "inputnode.target_img"),
                ("formatted_anat", "inputnode.name_source"),
                ]),
            (resample_template_node, anat_inho_cor_wf, [
                ("resampled_template", "inputnode.anat_ref"),
                ("resampled_mask", "inputnode.anat_mask"),
                ]),
            (resample_template_node, anat_inho_cor_wf, [
                ("resampled_template", "template_inputnode.template_anat"),
                ("resampled_mask", "template_inputnode.template_mask"),
                ]),
            (anat_inho_cor_wf, bold_main_wf, [
                ("outputnode.corrected", "inputnode.coreg_anat"),
                ]),
            (commonspace_reg_wf, bold_main_wf, [
                ("outputnode.native_mask", "inputnode.coreg_mask"),
                ("outputnode.unbiased_template", "template_inputnode.template_anat"),
                ("outputnode.unbiased_mask", "template_inputnode.template_mask"),
                ]),
            (EPI_target_buffer, bold_main_wf, [
                ("EPI_template", "inputnode.inho_cor_anat"),
                ("EPI_mask", "inputnode.inho_cor_mask"),
                ]),
            (anat_inho_cor_wf, commonspace_reg_wf, [
                ("outputnode.corrected", "inputnode.moving_image"),
                ("outputnode.denoise_mask", "inputnode.moving_mask"),
                ]),
            (anat_inho_cor_wf, EPI_target_buffer, [
                ("outputnode.corrected", "EPI_template"),
                ]),
            (commonspace_reg_wf, EPI_target_buffer, [
                ("outputnode.native_mask", 'EPI_mask'),
                ]),
            ])

        if not opts.anat_inho_cor['method']=='disable':
            anat_inho_cor_diagnosis = pe.Node(Function(input_names=['raw_img','init_denoise','warped_mask','final_denoise', 'name_source', 'out_dir', 'figure_format'],
                                               function=preprocess_visual_QC.inho_cor_diagnosis),
                                      name='anat_inho_cor_diagnosis')
            anat_inho_cor_diagnosis.inputs.out_dir = output_folder+'/preprocess_QC_report/anat_inho_cor/'
            anat_inho_cor_diagnosis.inputs.figure_format = opts.figure_format

            workflow.connect([
                (format_anat_buffer, anat_inho_cor_diagnosis, [
                    ("formatted_anat", "raw_img"),
                    ]),
                (anat_selectfiles, anat_inho_cor_diagnosis, [
                    ("out_file", "name_source"),
                    ]),
                (anat_inho_cor_wf, anat_inho_cor_diagnosis, [
                    ("outputnode.init_denoise", "init_denoise"),
                    ("outputnode.corrected", "final_denoise"),
                    ("outputnode.denoise_mask", "warped_mask"),
                    ]),
                ])

    else:
        inho_cor_bold_main_wf = init_bold_main_wf(
            output_folder=output_folder, number_functional_scans=number_functional_scans, inho_cor_only=True, name='inho_cor_bold_main_wf', opts=opts)

        workflow.connect([
            (resample_template_node, inho_cor_bold_main_wf, [
                ("resampled_template", "template_inputnode.template_anat"),
                ("resampled_mask", "template_inputnode.template_mask"),
                ]),
            (format_bold_buffer, inho_cor_bold_main_wf, [
                ("formatted_bold", "inputnode.bold"),
                ]),
            (EPI_target_buffer, inho_cor_bold_main_wf, [
                ("EPI_template", "inputnode.inho_cor_anat"),
                ("EPI_mask", "inputnode.inho_cor_mask"),
                ]),
            (inho_cor_bold_main_wf, bold_main_wf, [
                ("transitionnode.bold_file", "transitionnode.bold_file"),
                ("transitionnode.isotropic_bold_file", "transitionnode.isotropic_bold_file"),
                ("transitionnode.bold_ref", "transitionnode.bold_ref"),
                ("transitionnode.init_denoise", "transitionnode.init_denoise"),
                ("transitionnode.denoise_mask", "transitionnode.denoise_mask"),
                ("transitionnode.corrected_EPI", "transitionnode.corrected_EPI"),
                ]),
            (inho_cor_bold_main_wf, commonspace_reg_wf, [
                ("transitionnode.corrected_EPI", "inputnode.moving_image"),
                ("transitionnode.denoise_mask", "inputnode.moving_mask"),
                ]),
            (resample_template_node, EPI_target_buffer, [
                ("resampled_template", "EPI_template"),
                ("resampled_mask", "EPI_mask"),
                ]),
            ])

    if not opts.bold_only:
        PlotOverlap_EPI2Anat_node = pe.Node(
            preprocess_visual_QC.PlotOverlap(), name='PlotOverlap_EPI2Anat')
        PlotOverlap_EPI2Anat_node.inputs.out_dir = output_folder+'/preprocess_QC_report/EPI2Anat'
        workflow.connect([
            (bold_selectfiles, PlotOverlap_EPI2Anat_node,
             [("out_file", "name_source")]),
            (anat_inho_cor_wf, PlotOverlap_EPI2Anat_node,
             [("outputnode.corrected", "fixed")]),
            (outputnode, PlotOverlap_EPI2Anat_node, [
                ("inho_cor_bold_warped2anat", "moving"),  # warped EPI to anat
                ]),
            ])

    # fill the datasinks
    workflow.connect([
        (bold_selectfiles, bold_datasink, [
            ("out_file", "input_bold"),
            ]),
        (outputnode, motion_datasink, [
            ("motion_params_csv", "motion_params_csv"),  # confounds file
            ("FD_voxelwise", "FD_voxelwise"),
            ("pos_voxelwise", "pos_voxelwise"),
            ("FD_csv", "FD_csv"),
            ]),
        (outputnode, bold_datasink, [
            ("initial_bold_ref", "initial_bold_ref"),  # inspect initial bold ref
            ("inho_cor_bold", "inho_cor_bold"),  # inspect bias correction
            ("native_brain_mask", "native_brain_mask"),  # get the EPI labels
            ("native_WM_mask", "native_WM_mask"),  # get the EPI labels
            ("native_CSF_mask", "native_CSF_mask"),  # get the EPI labels
            ("native_vascular_mask", "native_vascular_mask"),  # get the EPI labels
            ("native_labels", "native_labels"),  # get the EPI labels
            # warped EPI to anat
            ("inho_cor_bold_warped2anat", "inho_cor_bold_warped2anat"),
            # resampled EPI after motion realignment and SDC
            ("native_bold", "native_bold"),
            # resampled EPI after motion realignment and SDC
            ("native_bold_ref", "native_bold_ref"),
            # resampled EPI after motion realignment and SDC
            ("commonspace_bold", "commonspace_bold"),
            ("commonspace_mask", "commonspace_mask"),
            ("commonspace_WM_mask", "commonspace_WM_mask"),
            ("commonspace_CSF_mask", "commonspace_CSF_mask"),
            ("commonspace_vascular_mask", "commonspace_vascular_mask"),
            ("commonspace_labels", "commonspace_labels"),
            ("tSNR_filename", "tSNR_map_preprocess"),
            ("std_filename", "std_map_preprocess"),
            ("commonspace_resampled_template", "commonspace_resampled_template"),
            ("raw_brain_mask", "raw_brain_mask"),
            ]),
        ])

    if not opts.bold_only:
        workflow.connect([
            (anat_inho_cor_wf, anat_datasink, [
                ("outputnode.corrected", "anat_preproc"),
                ]),
            (outputnode, transforms_datasink, [
                ('bold_to_anat_affine', 'bold_to_anat_affine'),
                ('bold_to_anat_warp', 'bold_to_anat_warp'),
                ('bold_to_anat_inverse_warp', 'bold_to_anat_inverse_warp'),
                ]),
            ])

    if opts.PEDir:
        pe_split = pe.Node(niu.IdentityInterface(fields=['pe', 'split_name']),
                            name="pe_split")
        pe_split.itersource = ('main_split', 'split_name')
        pe_split.iterables = ('pe', pe_iter)

        workflow.connect([
            (main_split, pe_split, [
                ("split_name", "split_name"),
            ]),
            (pe_split, bold_selectfiles, [
                ("pe", "pe"),
            ]),
            ])
    return workflow
