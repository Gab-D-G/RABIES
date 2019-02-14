EPI=$1
anat_file=$2
mask=$3

antsRegistration -d 3 \
--verbose -o [output_,output_warped_image.nii.gz] \
-t Rigid[0.1] -m Mattes[$anat_file,$EPI,1,64,None] \
-c 1000x500x250x100x50x25 -s 8x4x2x1x0.5x0 -f 6x5x4x3x2x1 --masks [NULL,NULL] \
-t Similarity[0.1] -m Mattes[$anat_file,$EPI,1,64,None] \
-c 100x50x25 -s 1x0.5x0 -f 3x2x1 --masks [$mask,NULL] \
-t Affine[0.1] -m Mattes[$anat_file,$EPI,1,64,None] \
-c 100x50x25 -s 1x0.5x0 -f 3x2x1 --masks [$mask,$mask] \
-t SyN[ 0.2, 3.0, 0.0 ] -m Mattes[$anat_file,$EPI, 1, 64 ] \
-c [ 40x20x0, 1e-06, 6 ] -s 2x1x0 -f 4x2x1 --masks [$mask,$mask] \
--interpolation BSpline[5] -z 1 -u 0 -a 1
