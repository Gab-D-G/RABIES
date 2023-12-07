import os
import pathlib
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import Function

from .analysis_functions import run_group_ICA, run_DR_ICA, run_FC_matrix, seed_based_FC


def init_analysis_wf(opts, commonspace_cr=False, name="analysis_wf"):

    workflow = pe.Workflow(name=name)
    subject_inputnode = pe.Node(niu.IdentityInterface(
        fields=['dict_file', 'token']), name='subject_inputnode')
    group_inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file_list', 'commonspace_mask', 'token']), name='group_inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['group_ICA_dir', 'IC_file', 'dual_regression_timecourse_csv',
                                                       'DR_nii_file', 'matrix_data_file', 'matrix_fig', 'corr_map_file', 'seed_timecourse_csv', 'joined_corr_map_file', 'joined_seed_timecourse_csv',
                                                       'sub_token', 'group_token','NPR_prior_timecourse_csv', 'NPR_extra_timecourse_csv',
                                                       'NPR_prior_filename', 'NPR_extra_filename', 'NPR_optimize_report']), name='outputnode')

    # connect the nodes so that they exist even without running analysis
    workflow.connect([
        (subject_inputnode, outputnode, [
            ("token", "sub_token"),
            ]),
        (group_inputnode, outputnode, [
            ("token", "group_token"),
            ]),
        ])

    if len(opts.seed_list) > 0:
        if not commonspace_cr:
            raise ValueError(
                'Outputs from confound regression must be in commonspace to run seed-based analysis. Try running confound regression again without --nativespace_analysis.')
        seed_based_FC_node = pe.Node(Function(input_names=['dict_file', 'seed_dict', 'seed_name'],
                                              output_names=['corr_map_file', 'seed_timecourse_csv'],
                                              function=seed_based_FC),
                                     name='seed_based_FC', mem_gb=1*opts.scale_min_memory)
        seed_dict = {}
        name_list = []
        for file in opts.seed_list:
            file = os.path.abspath(file)
            if not os.path.isfile(file):
                raise ValueError(
                    f"Provide seed file path {file} doesn't exists.")
            seed_name = pathlib.Path(file).name.rsplit(".nii")[0]
            name_list.append(seed_name)
            seed_dict[seed_name] = file
        seed_based_FC_node.iterables = ('seed_name', name_list)
        seed_based_FC_node.inputs.seed_dict = seed_dict

        # create a joinnode to provide also combined seed maps
        seed_FC_joinnode = pe.JoinNode(niu.IdentityInterface(fields=['joined_corr_map_file', 'joined_seed_timecourse_csv']),
                                                name='seed_FC_joinnode',
                                                joinsource='seed_based_FC',
                                                joinfield=['joined_corr_map_file', 'joined_seed_timecourse_csv'])


        workflow.connect([
            (subject_inputnode, seed_based_FC_node, [
                ("dict_file", "dict_file"),
                ]),
            (seed_based_FC_node, outputnode, [
                ("corr_map_file", "corr_map_file"),
                ("seed_timecourse_csv", "seed_timecourse_csv"),
                ]),
            (seed_based_FC_node, seed_FC_joinnode, [
                ("corr_map_file", "joined_corr_map_file"),
                ("seed_timecourse_csv", "joined_seed_timecourse_csv"),
                ]),
            (seed_FC_joinnode, outputnode, [
                ("joined_corr_map_file", "joined_corr_map_file"),
                ("joined_seed_timecourse_csv", "joined_seed_timecourse_csv"),
                ]),
            ])

    if opts.DR_ICA or opts.data_diagnosis:

        DR_ICA = pe.Node(Function(input_names=['dict_file', 'network_weighting'],
                                  output_names=['DR_maps_filename', 'dual_regression_timecourse_csv'],
                                  function=run_DR_ICA),
                         name='DR_ICA', mem_gb=1*opts.scale_min_memory)
        DR_ICA.inputs.network_weighting = opts.network_weighting

        workflow.connect([
            (subject_inputnode, DR_ICA, [
                ("dict_file", "dict_file"),
                ]),
            (DR_ICA, outputnode, [
                ("dual_regression_timecourse_csv", "dual_regression_timecourse_csv"),
                ("DR_maps_filename", "DR_nii_file"),
                ]),
            ])


    if opts.group_ica['apply']:
        if not commonspace_cr:
            raise ValueError(
                'Outputs from confound regression must be in commonspace to run group-ICA. Try running confound regression again without --nativespace_analysis.')
        group_ICA = pe.Node(Function(input_names=['bold_file_list', 'mask_file', 'dim', 'random_seed'],
                                     output_names=['out_dir', 'IC_file'],
                                     function=run_group_ICA),
                            name='group_ICA', mem_gb=1*opts.scale_min_memory)
        group_ICA.inputs.dim = opts.group_ica['dim']
        group_ICA.inputs.random_seed = opts.group_ica['random_seed']

        workflow.connect([
            (group_inputnode, group_ICA, [
                ("bold_file_list", "bold_file_list"),
                ("commonspace_mask", "mask_file"),
                ]),
            (group_ICA, outputnode, [
                ("IC_file", "IC_file"),
                ("out_dir", "group_ICA_dir"),
                ]),
            ])

    if (opts.NPR_temporal_comp>-1) or (opts.NPR_spatial_comp>-1) or opts.optimize_NPR['apply']:
        from .analysis_functions import NeuralPriorRecovery

        NPR_node = pe.Node(NeuralPriorRecovery(
                            NPR_temporal_comp=opts.NPR_temporal_comp, 
                            NPR_spatial_comp=opts.NPR_spatial_comp, 
                            optimize_NPR_dict=opts.optimize_NPR, 
                            prior_bold_idx = opts.prior_bold_idx,
                            network_weighting=opts.network_weighting,
                            figure_format=opts.figure_format,
                            CPCA_DR=opts.CPCA_DR),
                            name='NPR_node', mem_gb=1*opts.scale_min_memory)

        workflow.connect([
            (subject_inputnode, NPR_node, [
                ("dict_file", "dict_file"),
                ]),
            (NPR_node, outputnode, [
                ("NPR_prior_timecourse_csv", "NPR_prior_timecourse_csv"),
                ("NPR_extra_timecourse_csv", "NPR_extra_timecourse_csv"),
                ("NPR_prior_filename", "NPR_prior_filename"),
                ("NPR_extra_filename", "NPR_extra_filename"),
                ]),
            ])
        
    if opts.optimize_NPR['apply']:
        workflow.connect([
            (NPR_node, outputnode, [
                ("optimize_report", "NPR_optimize_report"),
                ]),
            ])


    if opts.FC_matrix:
        FC_matrix = pe.Node(Function(input_names=['dict_file', 'figure_format', 'roi_type'],
                                     output_names=['data_file', 'figname'],
                                     function=run_FC_matrix),
                            name='FC_matrix', mem_gb=1*opts.scale_min_memory)
        FC_matrix.inputs.roi_type = opts.ROI_type
        FC_matrix.inputs.figure_format = opts.figure_format

        workflow.connect([
            (subject_inputnode, FC_matrix, [
                ("dict_file", "dict_file"),
                ]),
            (FC_matrix, outputnode, [
                ("data_file", "matrix_data_file"),
                ("figname", "matrix_fig"),
                ]),
            ])

    return workflow
