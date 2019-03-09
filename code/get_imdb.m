function [imdb, im_files] = get_imdb( train_filenames, data_path, im_size, im_crop, im_transform )

    % image transform
    if nargin<5
        im_transform = @(im)(permute(single(im), [4 3 1 2])/127.5 - 1);
    end

    % load
    im_files = textread(train_filenames, '%s\n');
    if nargin<4 || isempty(im_crop)
        imdb.images = single(zeros(numel(im_files), 3, im_size(1), im_size(2)));
    else
        imdb.images = single(zeros(numel(im_files), ...
                                   3,...
                                   im_crop(3)-im_crop(1)+1, ...
                                   im_crop(4)-im_crop(2)+1));
    end
    for i = 1:numel(im_files)
        im = imread(fullfile(data_path, im_files{i}));
        if size(im,3)==1
            im = repmat(im, [1 1 3]);
        end
        im = im(im_crop(1):im_crop(3), im_crop(2):im_crop(4), :);
        imdb.images(i,:,:,:) = im_transform(im);
    end

    % shuffle
    permutation = randperm(size(imdb.images,1));
    imdb.images = imdb.images(permutation, :, :, :);
    im_files = im_files(permutation);
    
end