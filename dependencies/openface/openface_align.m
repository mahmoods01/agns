function [ims_out, tforms] = openface_align(ims_in, align_info, default_src_pts)

    n = size(ims_in,4);
    prefix = 'face_';
    filenames = repmat({''}, [n 1]);
    landmarks_filenames = repmat({''}, [n 1]);
    
    if nargin<3
        default_src_pts = [];
    end
    
    % write images to tmp files
    for i_im = 1:n
        im = ims_in(:,:,:,i_im);
        filename = [tempname('/') '.png'];
        filename = [prefix filename(2:end)];
        landmarks_filename = ['Fast_Marks_' filename '.csv'];
        imwrite(im, fullfile('/tmp/', filename));
        filenames{i_im} = filename;
        landmarks_filenames{i_im} = landmarks_filename;
    end
    
    % run dlib's landmark detection via python
    python = '<UPDATE ME>'; % Full path to the python executable
    dlib = '<UPDATE ME>'; % Full path to dlib's 'shape_predictor_68_face_landmarks.dat'
    command_str = [python ' ' ...
                   fullfile(pwd, 'dependencies/image-registration/face_landmark_detection2.py ')...
                   dlib ' '...
                   '/tmp/ face_*.png'];
    system(command_str);

    
    % read the files and perform the alignment
    ims_out = single(zeros(align_info.dim, align_info.dim, 3, n));
    tforms = repmat({[]}, [n 1]);
    for i_im = 1:n
        im = ims_in(:,:,:,i_im);
        filename = filenames{i_im};
        landmarks_filename = landmarks_filenames{i_im};
        if exist(fullfile('/tmp/', landmarks_filename), 'file')
            % if landmarks found, find transform
            landmarks = csvread(fullfile('/tmp/', landmarks_filename));
            source_pts = landmarks(align_info.indices, :);
            target_pts = align_info.points(align_info.indices, :);
        elseif ~isempty(default_src_pts)
            % if no landmarks found, and default src pts are provided
            source_pts = default_src_pts;
            target_pts = align_info.points(align_info.indices, :);
        else
            % if no landmarks found, and no default src pts are provded,
            % find resize transform
            source_pts = [1 1; size(im,1) 1; 1 size(im,2)];
            target_pts = [1 1; align_info.dim 1; 1 align_info.dim];
        end
        tform = fitgeotrans(source_pts, target_pts, 'affine');
        im = imwarp(im, tform, 'OutputView', imref2d([align_info.dim align_info.dim]),...
                    'Interp', 'nearest');
        ims_out(:,:,:,i_im) = im;
        tforms{i_im} = tform;
        % clean
        delete(fullfile('/tmp/', filename));
        if exist(fullfile('/tmp/', landmarks_filename), 'file')
            delete(fullfile('/tmp/', landmarks_filename));
        end
    end
   
    
end
