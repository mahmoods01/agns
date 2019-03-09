% This demo shows examples of digital and physical attacks via AGNs against
% the VGG and OpeFace DNNs.

%% initialize paths to attack code, dependencies, ...
run init.m

%% Brad Pitt (subject #19) digital dodging vs. VGG143

% parameters
n_epochs = 1;
stop_prob = 0.01;
kappa = 0.25;
lr = 5e-5;
weight_decay = 1e-5;

% cpu/gpu?
platform = 'cpu';
if gpuDeviceCount>0
    platform = 'gpu';
end

% model and attacker images
face_net_path = 'models/vgg143-recognition-nn.mat';
im_filepath = {'data/demo-data1/aligned_vgg_brad_pitt.jpg'};

% attacker's class
target = 19;

% run attack
[gen, discrim, objective] = agn('kappa', kappa, ...
                                'lr_gen', lr, ...
                                'weight_decay', weight_decay, ...
                                'attack', 'dodge', ...
                                'targets', target, ...
                                'stop_prob', stop_prob, ...
                                'face_im_paths', im_filepath, ...
                                'n_epochs', n_epochs, ...
                                'platform', platform, ...
                                'face_net_path', face_net_path ...
                                );

% store results
result = struct('gen', gen, 'discrim', discrim, 'objective', objective, ...
                'target', target, 'im_path', im_filepath,  ...
                'attack_type', 'dodge', 'stop_prob', stop_prob, ... 
                'lr', lr, 'kappa', kappa, 'face_net_path', face_net_path, ...
                'weight_decay', weight_decay);
save('results/demo-digital-dodge-vgg143.mat', 'result');

%% digital impersonation vs. Openface10

% parameters
n_epochs = 1;
stop_prob = 0.924;
kappa = 0.25;
lr = 5e-5;
weight_decay = 1e-5;

% cpu/gpu?
platform = 'cpu';
if gpuDeviceCount>0
    platform = 'gpu';
end

% model and attacker's image
face_net_path = 'models/openface10-recognition-nn.mat';
im_filepath = {'data/demo-data1/aligned_vgg_brad_pitt.jpg'};

% target to impersonate (Aaron Eckhart)
target = 1;

% find transforms to align images & eyeglasses to the openface pose
align_data = load('dependencies/openface/alignment_info.mat');
im = imread(im_filepath{1});
[~, tforms] = openface_align(im, align_data.align_info);
alignment.tforms = tforms{1};
alignment.im_size = [align_data.align_info.dim align_data.align_info.dim];

% run attack
[gen, discrim, objective] = agn_dagnn(...
                                      'kappa', kappa, ...
                                      'lr_gen', lr, ...
                                      'weight_decay', weight_decay, ...
                                      'attack', 'impersonate', ...
                                      'targets', target, ...
                                      'stop_prob', stop_prob, ...
                                      'face_im_paths', im_filepath, ...
                                      'n_epochs', n_epochs, ...
                                      'platform', platform, ...
                                      'face_net_path', face_net_path, ...
                                      'alignment', alignment ...
                                      );
                                  
% store results
result = struct('gen', gen, 'discrim', discrim, 'objective', objective, ...
                'target', target, 'im_path', im_filepath,  ...
                'attack_type', 'dodge', 'stop_prob', stop_prob, ... 
                'lr', lr, 'kappa', kappa, 'face_net_path', face_net_path, ...
                'weight_decay', weight_decay, 'alignment', alignment);
save('results/demo-digital-impersonation-openface10.mat', 'result');

%% physical impersonation vs. VGG10

% parameters
n_epochs = 1;
n_ims = 6;
stop_prob = 0.924;
kappa = 0.25;
lr = 5e-5;
weight_decay = 1e-5;

% gpu/cpu?
platform = 'cpu';
if gpuDeviceCount>0
    platform = 'gpu';
end
             
% model and attacker's images
face_net_path = 'models/vgg10-recognition-nn.mat';
im_wildcard = 'aligned_*.png';
im_dir = 'data/demo-data2/';
im_files = dir(fullfile(im_dir, im_wildcard));
im_files = im_files(randperm(numel(im_files)));
im_files = im_files(1:min([n_ims numel(im_files)]));
im_filepath = fullfile(im_dir, extractfield(im_files, 'name'));

% target to impersonate
target = 6;
targets = repmat(target, [numel(im_filepath) 1]);

% transform eyeglasses based on the green marks
load('data/auxiliary/eyeglass_marks_centers.mat', 'eyeglass_marks_centers');
eyeglass_tforms = [];
for i_m = 1:numel(im_filepath)
    im = imread(im_filepath{i_m});
    centers = find_green_marks(im, 0, [155 155 155]);
    tform = fitgeotrans(eyeglass_marks_centers, centers, 'projective');
    eyeglass_tforms = cat(1, eyeglass_tforms, tform);
end

% run attack
[gen, discrim, objective] = agn_realizable(...
                                           'face_net_path', face_net_path, ...
                                           'kappa', kappa, ...
                                           'lr_gen', lr, ...
                                           'weight_decay', weight_decay, ...
                                           'attack', 'impersonate', ...
                                           'targets', targets, ...
                                           'stop_prob', stop_prob, ...
                                           'face_im_paths', im_filepath, ...
                                           'eyeglass_tforms', eyeglass_tforms, ...
                                           'n_epochs', n_epochs, ...
                                           'platform', platform ...
                                           );

% store results
result  = struct('gen', gen, 'discrim', discrim, 'objective', objective, ...
                 'target', targets, 'im_path', im_filepath, 'attack_type', 'impersonate', ...
                 'stop_prob', stop_prob, 'lr', lr, 'kappa', kappa, ...
                 'weight_decay', weight_decay, 'face_net_path', '', ...
                 'eyeglass_tforms', eyeglass_tforms);
save('results/demo-physical-imperonation-vgg10.mat', 'result');

%% physical dodging vs. Openface143

% parameters
n_epochs = 1;
n_ims = 32;
stop_prob = 0.01;
kappa = 0.25;
lr = 5e-5;
weight_decay = 1e-5;

% cpu/gpu?
platform = 'cpu';
if gpuDeviceCount>0
    platform = 'gpu';
end

% model and attacker's images
face_net_path = 'models/openface143-recognition-nn.mat';
im_wildcard = 'aligned_*.png';
im_dir = 'data/demo-data2/';
im_files = dir(fullfile(im_dir, im_wildcard));
im_files = im_files(randperm(numel(im_files)));
im_files = im_files(1:min([n_ims numel(im_files)]));
im_filepath = fullfile(im_dir, extractfield(im_files, 'name'));

% attacker's class
target = 142;
targets = repmat(target, [numel(im_filepath) 1]);

% transform eyeglasses based on the green marks
load('data/auxiliary/eyeglass_marks_centers.mat', 'eyeglass_marks_centers');
eyeglass_tforms = [];
for i_m = 1:numel(im_filepath)
    im = imread(im_filepath{i_m});
    centers = find_green_marks(im, 0, [155 155 155]);
    tform = fitgeotrans(eyeglass_marks_centers, centers, 'projective');
    eyeglass_tforms = cat(1, eyeglass_tforms, tform);
end

% find transforms to align images & eyeglasses to the openface pose
ims = [];
for i_im = 1:numel(im_filepath)
    ims = cat(4, ims, imread(im_filepath{i_im}));
end
align_data = load('dependencies/openface/alignment_info.mat');
[~, tforms] = openface_align(ims, align_data.align_info);
alignment.tforms = [];
for i_t = 1:numel(tforms)
    alignment.tforms = cat(1, alignment.tforms, tforms{i_t});
end
alignment.im_size = [align_data.align_info.dim align_data.align_info.dim];

% run attack
[gen, discrim, objective] = agn_realizable_dagnn(...
                                            'face_net_path', face_net_path, ...
                                            'alignment', alignment, ...
                                            'kappa', kappa, ...
                                            'lr_gen', lr, ...
                                            'weight_decay', weight_decay, ...
                                            'attack', 'dodge', ...
                                            'targets', targets, ...
                                            'stop_prob', stop_prob, ...
                                            'face_im_paths', im_filepath, ...
                                            'n_epochs', n_epochs, ...
                                            'platform', platform, ...
                                            'eyeglass_tforms', eyeglass_tforms ...
                                            );

% store results
result  = struct('gen', gen, 'discrim', discrim, 'objective', objective, ...
                 'target', targets, 'im_path', im_filepath, 'attack_type', 'dodge', ...
                 'stop_prob', stop_prob, 'lr', lr, 'kappa', kappa, ...
                 'weight_decay', weight_decay, 'face_net_path', '', ...
                 'eyeglass_tforms', eyeglass_tforms, 'alignment', alignment);
save('results/demo-physical-dodege-openface143.mat', 'result');
