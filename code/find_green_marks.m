function centers = find_green_marks(im, verbose, rgb_t)

    if nargin<3
        r_t = 130;  % red threshold
        g_t = 140;   % green threshold
        b_t = 160;   % blue threshold
    else
        r_t = rgb_t(1);
        g_t = rgb_t(2);
        b_t = rgb_t(3);
    end

    % binarize image
    im = 1.*( im(:,:,1)<r_t & im(:,:,2)>g_t & im(:,:,3)<b_t );

    % find connected components
    cc = bwconncomp(im);

    % there should be at least 4 components
    if numel(cc.PixelIdxList)<4
        centers = [];
        return;
    end

    % keep largest 7 components
    sizes = [];
    for i = 1:numel(cc.PixelIdxList)
        pixel_list = cc.PixelIdxList(i);
        sizes = cat(2, sizes, numel(pixel_list{1}));
    end
    [~, ix] = sort(sizes, 'descend');
    cc.PixelIdxList = cc.PixelIdxList(ix(1:7));

    % find the center of each area
    centers = [];
    for i = 1:numel(cc.PixelIdxList)
        pixel_list = cc.PixelIdxList(i);
        pixel_list = pixel_list{1};
        idxs = [];
        for j = 1:numel(pixel_list)
            [r,c] = ind2sub(size(im), pixel_list(j));
            idxs = cat(1, idxs, [r c]);
        end
        centers = cat(1, centers, round(mean(idxs, 1)));
    end
    
    % order centes (3 top marks from left to right, followed by 4 bottom
    % marks from left to right)
    [~,ix] = sort(centers(:,1));
    centers = centers(ix,:);
    [~, ix] = sort(centers(1:3,2));
    centers(1:3,:) = centers(ix,:);
    [~, ix] = sort(centers(4:7,2));
    centers(4:7,:) = centers(ix+3,:);
    
    % flip column's order
    centers = centers(:, [2 1]);

    % plot
    if nargin>1 && verbose
        imshow(im); hold on;
        for i = 1:size(centers,1)
            rectangle('Position', [centers(i,1)-1 centers(i,2)-1 3 3], 'EdgeColor', 'r');
        end
    end

end