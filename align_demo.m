% A demo for how to align 'data/demo-data1/brad_pitt.jpg'
addpath('dependencies/image-registration/');
face_landmark_detection('data/demo-data1/', 'brad_pitt.jpg');
align_vgg_pose('data/demo-data1/', 'brad_pitt.jpg', ...
               'data/auxiliary/canonical_pose_marks.csv');
