import os
import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, InputMultiPath, BaseInterface, SimpleInterface
)
from nipype.interfaces.base import CommandLine, CommandLineInputSpec

def prep_bids_iter(layout):
    '''
    This function takes as input a BIDSLayout, and returns a list of subjects with
    their associated number of sessions and runs.
    '''
    subject_list=layout.get_subject()
    #create a dictionary with list of bold session and run numbers for each subject
    session_iter={}
    run_iter={}
    for sub in subject_list:
        sub_func=layout.get(subject=sub, datatype='func', extension=['nii', 'nii.gz'])
        session=0
        run=0
        for func_bids in sub_func:
            if int(func_bids.get_entities()['session'])>session:
                session=int(func_bids.get_entities()['session'])
            if int(func_bids.get_entities()['run'])>run:
                run=int(func_bids.get_entities()['run'])
        session_iter[sub] = list(range(1,int(session)+1))
        run_iter[sub] = list(range(1,int(run)+1))
    return subject_list, session_iter, run_iter

class BIDSDataGraberInputSpec(BaseInterfaceInputSpec):
    bids_dir = traits.Str(exists=True, mandatory=True, desc="BIDS data directory")
    datatype = traits.Str(exists=True, mandatory=True, desc="datatype of the target file")
    subject_id = traits.Str(exists=True, mandatory=True, desc="Subject ID")
    session = traits.Int(exists=True, mandatory=True, desc="Session number")
    run = traits.Int(exists=True, default=None, desc="Run number")

class BIDSDataGraberOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Selected file based on the provided parameters.")

class BIDSDataGraber(BaseInterface):
    """
    This interface will select a single scan from the BIDS directory based on the
    input specifications.
    """

    input_spec = BIDSDataGraberInputSpec
    output_spec = BIDSDataGraberOutputSpec

    def _run_interface(self, runtime):
        import os
        from bids.layout import BIDSLayout
        layout = BIDSLayout(self.inputs.bids_dir)
        try:
            if self.inputs.datatype=='func':
                bids_file=layout.get(subject=self.inputs.subject_id, session=self.inputs.session, run=self.inputs.run, extension=['nii', 'nii.gz'], datatype=self.inputs.datatype)
                func=layout.get(subject=self.inputs.subject_id, session=self.inputs.session, run=self.inputs.run, extension=['nii', 'nii.gz'], datatype=self.inputs.datatype, return_type='filename')
                file=func[0]
            elif self.inputs.datatype=='anat':
                bids_file=layout.get(subject=self.inputs.subject_id, session=self.inputs.session, extension=['nii', 'nii.gz'], datatype=self.inputs.datatype)
                anat=layout.get(subject=self.inputs.subject_id, session=self.inputs.session, extension=['nii', 'nii.gz'], datatype=self.inputs.datatype, return_type='filename')
                file=anat[0]
            else:
                raise ValueError('Wrong datatype %s' % (self.inputs.datatype))
            if len(bids_file)>1:
                raise ValueError('Provided BIDS spec lead to duplicates: %s' % (str(self.inputs.datatype+'_'+self.inputs.subject_id+'_'+self.inputs.session+'_'+self.inputs.run)))
        except:
            raise ValueError('Error with BIDS spec: %s' % (str(self.inputs.datatype+'_'+self.inputs.subject_id+'_'+self.inputs.session+'_'+self.inputs.run)))

        #RABIES only work with compressed .nii for now
        if not '.nii.gz' in file:
            print('Compressing BIDS input to .gz')
            command='gzip %s' % (file,)
            if os.system(command) != 0:
                raise ValueError('Error in '+command)
            file=file+'.gz'

        setattr(self, 'out_file', file)

        return runtime

    def _list_outputs(self):
        return {'out_file': getattr(self, 'out_file')}


def init_bold_reference_wf(detect_dummy=False, name='gen_bold_ref'):
    """
    This workflow generates reference BOLD images for a series

    **Parameters**

        detect_dummy : bool
            whether to detect and remove dummy volumes, and generate a BOLD ref
            volume based on the contrast enhanced dummy volumes.
        name : str
            Name of workflow (default: 'gen_bold_ref')

    **Inputs**

        bold_file
            BOLD series NIfTI file

    **Outputs**

        bold_file
            Validated BOLD series NIfTI file
        ref_image
            Reference image generated by taking the median from the motion-realigned BOLD timeseries
        skip_vols
            Number of non-steady-state volumes detected at beginning of ``bold_file``
        validation_report
            HTML reportlet indicating whether ``bold_file`` had a valid affine

    """
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from .interfaces import ValidateImage


    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'skip_vols', 'ref_image', 'validation_report']),
        name='outputnode')


    '''
    Check the correctness of x-form headers (matrix and code)

        This interface implements the `following logic
        <https://github.com/poldracklab/fmriprep/issues/873#issuecomment-349394544>
    '''
    validate = pe.Node(ValidateImage(), name='validate', mem_gb=2)
    validate.plugin_args = {'qsub_args': '-pe smp %s' % (str(2*int(os.environ["min_proc"]))), 'overwrite': True}

    gen_ref = pe.Node(EstimateReferenceImage(detect_dummy=detect_dummy), name='gen_ref', mem_gb=2)
    gen_ref.plugin_args = {'qsub_args': '-pe smp %s' % (str(2*int(os.environ["min_proc"]))), 'overwrite': True}

    workflow.connect([
        (inputnode, validate, [('bold_file', 'in_file')]),
        (validate, gen_ref, [('out_file', 'in_file')]),
        (validate, outputnode, [('out_file', 'bold_file'),
                                ('out_report', 'validation_report')]),
        (gen_ref, outputnode, [('ref_image', 'ref_image'),
                               ('n_volumes_to_discard', 'skip_vols')]),
    ])

    return workflow


class EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")
    detect_dummy = traits.Bool(desc="specify if should detect and remove dummy scans, and use these volumes as reference image.")

class EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(desc="Number of detected non-steady "
                                           "state volumes in the beginning of "
                                           "the input file")

class EstimateReferenceImage(BaseInterface):
    """
    Given a 4D EPI file, estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If the detect_dummy
    option is selected, it will use detected anat saturated volumes (non-steady
    state). Otherwise, a median of a subset of motion corrected volumes is used.
    In the later case, a first median is extracted from the raw data and used as
    reference for motion correction, then a new median image is extracted from
    the corrected series, and the process is repeated one more time to generate
    a final image reference image.
    """

    input_spec = EstimateReferenceImageInputSpec
    output_spec = EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):

        import os
        import nibabel as nb
        import numpy as np

        in_nii = nb.load(self.inputs.in_file)
        data_slice = in_nii.dataobj[:, :, :, :50]

        # Slicing may induce inconsistencies with shape-dependent values in extensions.
        # For now, remove all. If this turns out to be a mistake, we can select extensions
        # that don't break pipeline stages.
        in_nii.header.extensions.clear()

        n_volumes_to_discard = _get_vols_to_discard(in_nii)

        subject_id=os.path.basename(self.inputs.in_file).split('_ses-')[0]
        session=os.path.basename(self.inputs.in_file).split('_ses-')[1][0]
        run=os.path.basename(self.inputs.in_file).split('_run-')[1][0]
        filename_template = '%s_ses-%s_run-%s' % (subject_id, session, run)

        out_ref_fname = os.path.abspath('%s_bold_ref.nii.gz' % (filename_template))

        if (not n_volumes_to_discard == 0) and self.inputs.detect_dummy:
            print("Detected "+str(n_volumes_to_discard)+" dummy scans. Taking the median of these volumes as reference EPI.")
            median_image_data = np.median(
                data_slice[:, :, :, :n_volumes_to_discard], axis=3)
        else:
            print("Detected no dummy scans. Generating the ref EPI based on multiple volumes.")
            #if no dummy scans, will generate a median from a subset of max 40
            #slices of the time series
            if in_nii.shape[-1] > 40:
                slice_fname = os.path.abspath("slice.nii.gz")
                nb.Nifti1Image(data_slice[:, :, :, 20:40], in_nii.affine,
                               in_nii.header).to_filename(slice_fname)
                median_fname = os.path.abspath("median.nii.gz")
                nb.Nifti1Image(np.median(data_slice[:, :, :, 20:40], axis=3), in_nii.affine,
                               in_nii.header).to_filename(median_fname)
            else:
                slice_fname = self.inputs.in_file
                median_fname = os.path.abspath("median.nii.gz")
                nb.Nifti1Image(np.median(data_slice, axis=3), in_nii.affine,
                               in_nii.header).to_filename(median_fname)

            print("First iteration to generate reference image.")
            res = antsMotionCorr(in_file=slice_fname, ref_file=median_fname, second=False).run()
            median = np.median(nb.load(res.outputs.mc_corrected_bold).get_data(), axis=3)
            tmp_median_fname = os.path.abspath("tmp_median.nii.gz")
            nb.Nifti1Image(median, in_nii.affine,
                           in_nii.header).to_filename(tmp_median_fname)

            print("Second iteration to generate reference image.")
            res = antsMotionCorr(in_file=slice_fname, ref_file=tmp_median_fname, second=True).run()
            median_image_data = np.median(nb.load(res.outputs.mc_corrected_bold).get_data(), axis=3)

        #median_image_data is a 3D array of the median image, so creates a new nii image
        #saves it
        ref_img=nb.Nifti1Image(median_image_data, in_nii.affine,
                       in_nii.header)
        resample_image(ref_img, os.environ["rabies_data_type"]).to_filename(out_ref_fname)


        setattr(self, 'ref_image', out_ref_fname)
        setattr(self, 'n_volumes_to_discard', n_volumes_to_discard)

        return runtime

    def _list_outputs(self):
        return {'ref_image': getattr(self, 'ref_image'),
                'n_volumes_to_discard': getattr(self, 'n_volumes_to_discard')}


def _get_vols_to_discard(img):
    '''
    Takes a nifti file, extracts the mean signal of the first 50 volumes and computes which are outliers.
    is_outlier function: computes Modified Z-Scores (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) to determine which volumes are outliers.
    '''
    from nipype.algorithms.confounds import is_outlier
    data_slice = img.dataobj[:, :, :, :50]
    global_signal = data_slice.mean(axis=0).mean(axis=0).mean(axis=0)
    return is_outlier(global_signal)


class antsMotionCorrInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input BOLD time series')
    ref_file = File(exists=True, mandatory=True, desc='ref file to realignment time series')
    second = traits.Bool(desc="specify if it is the second iteration")

class antsMotionCorrOutputSpec(TraitedSpec):
    mc_corrected_bold = File(exists=True, desc="motion corrected time series")
    avg_image = File(exists=True, desc="average image of the motion corrected time series")
    csv_params = File(exists=True, desc="csv files with the 6-parameters rigid body transformations")

class antsMotionCorr(BaseInterface):
    """
    This interface performs motion realignment using antsMotionCorr function. It takes a reference volume to which
    EPI volumes from the input 4D file are realigned based on a Rigid registration.
    """

    input_spec = antsMotionCorrInputSpec
    output_spec = antsMotionCorrOutputSpec

    def _run_interface(self, runtime):

        import os
        #change the name of the first iteration directory to prevent overlap of files with second iteration
        if self.inputs.second:
            command='mv ants_mc_tmp first_ants_mc_tmp'
            if os.system(command) != 0:
                raise ValueError('Error in '+command)

        #make a tmp directory to store the files
        os.makedirs('ants_mc_tmp', exist_ok=True)

        command='antsMotionCorr -d 3 -o [ants_mc_tmp/motcorr,ants_mc_tmp/motcorr.nii.gz,ants_mc_tmp/motcorr_avg.nii.gz] \
                -m MI[ %s , %s , 1 , 20 , None ] -t Rigid[ 0.1 ] -i 100x50x30 -u 1 -e 1 -l 1 -s 2x1x0 -f 4x2x1 -n 10' % (self.inputs.ref_file,self.inputs.in_file)
        if os.system(command) != 0:
            raise ValueError('Error in '+command)

        setattr(self, 'csv_params', 'ants_mc_tmp/motcorrMOCOparams.csv')
        setattr(self, 'mc_corrected_bold', 'ants_mc_tmp/motcorr.nii.gz')
        setattr(self, 'avg_image', 'ants_mc_tmp/motcorr_avg.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'mc_corrected_bold': getattr(self, 'mc_corrected_bold'),
                'csv_params': getattr(self, 'csv_params'),
                'avg_image': getattr(self, 'avg_image')}


class slice_applyTransformsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input 4D EPI")
    ref_file = File(exists=True, mandatory=True, desc="The reference 3D space to which the EPI will be warped.")
    transforms = traits.List(desc="List of transforms to apply to every slice")
    inverses = traits.List(desc="Define whether some transforms must be inverse, with a boolean list where true defines inverse e.g.[0,1,0]")
    apply_motcorr = traits.Bool(default=True, desc="Whether to apply motion realignment.")
    motcorr_params = File(exists=True, desc="xforms from head motion estimation .csv file")
    resampling_dim = traits.Str(desc="Specification for the dimension of resampling.")
    data_type = traits.Str(default='float64', desc="Specify resampling data format to control for file size. Can specify a numpy data type from https://docs.scipy.org/doc/numpy/user/basics.types.html.")

class slice_applyTransformsOutputSpec(TraitedSpec):
    out_files = traits.List(desc="warped images after the application of the transforms")

class slice_applyTransforms(BaseInterface):
    """
    This interface will apply a set of transforms to an input 4D EPI as well as motion realignment if specified.
    Susceptibility distortion correction can be applied through the provided transforms. A list of the corrected
    single volumes will be provided as outputs, and these volumes require to be merged to recover timeseries.
    """

    input_spec = slice_applyTransformsInputSpec
    output_spec = slice_applyTransformsOutputSpec

    def _run_interface(self, runtime):
        #resampling the reference image to the dimension of the EPI
        from nibabel import processing
        import numpy as np
        import nibabel as nb
        import os
        img=nb.load(self.inputs.in_file)

        if not self.inputs.resampling_dim=='origin':
            resample_image(nb.load(self.inputs.ref_file), self.inputs.data_type, img_dim=self.inputs.resampling_dim).to_filename('resampled.nii.gz')
        else:
            shape=img.header.get_zooms()
            dims="%sx%sx%s" % (shape[0],shape[1],shape[2])
            resample_image(nb.load(self.inputs.ref_file), self.inputs.data_type, img_dim=dims).to_filename('resampled.nii.gz')

        #tranforms is a list of transform files, set in order of call within antsApplyTransforms
        transform_string=""
        for transform,inverse in zip(self.inputs.transforms, self.inputs.inverses):
            if bool(inverse):
                transform_string += "-t [%s,1] " % (transform,)
            else:
                transform_string += "-t %s " % (transform,)

        print("Splitting bold file into lists of single volumes")
        [bold_volumes, num_volumes] = split_volumes(self.inputs.in_file, "bold_")

        if self.inputs.apply_motcorr:
            motcorr_params=self.inputs.motcorr_params
        ref_img=os.path.abspath('resampled.nii.gz')
        warped_volumes = []
        for x in range(0, num_volumes):
            warped_vol_fname = os.path.abspath("deformed_volume" + str(x) + ".nii.gz")
            warped_volumes.append(warped_vol_fname)
            if self.inputs.apply_motcorr:
                command='antsMotionCorrStats -m %s -o motcorr_vol%s.mat -t %s' % (motcorr_params, x, x)
                if os.system(command) != 0:
                    raise ValueError('Error in '+command)
                command='antsApplyTransforms -i %s %s-t motcorr_vol%s.mat -n BSpline[5] -r %s -o %s' % (bold_volumes[x], transform_string, x, ref_img, warped_vol_fname)
                if os.system(command) != 0:
                    raise ValueError('Error in '+command)
            else:
                command='antsApplyTransforms -i %s %s-n BSpline[5] -r %s -o %s' % (bold_volumes[x], transform_string, ref_img, warped_vol_fname)
                if os.system(command) != 0:
                    raise ValueError('Error in '+command)
            #change image to specified data type
            img=nb.load(warped_vol_fname)
            img.set_data_dtype(self.inputs.data_type)
            nb.save(img, warped_vol_fname)

        setattr(self, 'out_files', warped_volumes)
        return runtime

    def _list_outputs(self):
        return {'out_files': getattr(self, 'out_files')}

def split_volumes(in_file, output_prefix):
    '''
    Takes as input a 4D .nii file and splits it into separate time series
    volumes by splitting on the 4th dimension
    '''
    import os
    import numpy as np
    import nibabel as nb
    in_nii = nb.load(in_file)
    num_dimensions = len(in_nii.shape)
    num_volumes = in_nii.shape[3]

    if num_dimensions!=4:
        print("the input file must be of dimensions 4")
        return None

    volumes = []
    for x in range(0, num_volumes):
        data_slice = in_nii.dataobj[:, :, :, x]
        slice_fname = os.path.abspath(output_prefix + "vol" + str(x) + ".nii.gz")
        nb.Nifti1Image(data_slice, in_nii.affine,
                       in_nii.header).to_filename(slice_fname)
        volumes.append(slice_fname)

    return [volumes, num_volumes]



class MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input list of files to merge, listed in the order to merge')
    header_source = File(exists=True, mandatory=True, desc='a Nifti file from which the header should be copied')
    data_type = traits.Str(default='float64', desc="Specify resampling data format to control for file size. Can specify a numpy data type from https://docs.scipy.org/doc/numpy/user/basics.types.html.")

class MergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output merged file')

class Merge(BaseInterface):
    """
    Takes a list of 3D Nifti files and merge them in the order listed.
    """

    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def _run_interface(self, runtime):
        import os
        import nibabel as nb
        import numpy as np

        subject_id=os.path.basename(self.inputs.header_source).split('_ses-')[0]
        session=os.path.basename(self.inputs.header_source).split('_ses-')[1][0]
        run=os.path.basename(self.inputs.header_source).split('_run-')[1][0]
        filename_template = '%s_ses-%s_run-%s' % (subject_id, session, run)

        img = nb.load(self.inputs.in_files[0])
        affine = img.affine
        header = nb.load(self.inputs.header_source).header
        length = len(self.inputs.in_files)
        combined = np.zeros((img.shape[0], img.shape[1], img.shape[2], length))

        i=0
        for file in self.inputs.in_files:
            combined[:,:,:,i] = nb.load(file).dataobj[:,:,:]
            i = i+1
        if (i!=length):
            print("Error occured with Merge.")
            return None
        combined_files = os.path.abspath("%s_combined.nii.gz" % (filename_template))
        combined_image=nb.Nifti1Image(combined, affine,
                       header)
        #change image to specified data type
        combined_image.set_data_dtype(self.inputs.data_type)
        nb.save(combined_image, combined_files)

        setattr(self, 'out_file', combined_files)
        return runtime

    def _list_outputs(self):
        return {'out_file': getattr(self, 'out_file')}



def resample_image(nb_image, data_type, img_dim='origin'):
    """
    This function takes as input a nibabel nifti image and changes the its data
    format as well as its voxel dimensions if specified.
    """
    import nibabel as nb
    if not img_dim=='origin':
        from nibabel import processing
        import numpy as np
        shape=img_dim.split('x')
        nb_image=processing.resample_to_output(nb_image, voxel_sizes=(float(shape[0]),float(shape[1]),float(shape[2])), order=4)
    nb_image.set_data_dtype(data_type)
    return nb_image
