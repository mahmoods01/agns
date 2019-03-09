function [y, dzdw, dzdb] = vl_nndeconv(x, w, b, pad, stride, mode, dzdy)
% An implementation of deconolution. A.k.a transposed convolution and 
% fractionally strided convolution.
% For more info: "A guide to convolution arithmetic for deep learning"

    dx = permute(x, [3 4 2 1]);
    w = permute(w, [3 4 2 1]);
    
    dzdw = []; dzdb = [];
    
    % backward
    if nargin>6
        x = permute(x, [3 4 2 1]);
        dzdy = permute(dzdy, [3 4 2 1]);
        w = w(end:-1:1, end:-1:1, :, :);
        [y, dzdw, dzdb] = vl_nnconvt(x, w, b, dzdy, 'upsample', stride, 'crop', [ceil(size(w,1)/4) floor(size(w,1)/4) ceil(size(w,2)/4) floor(size(w,2)/4)]);
        y = permute(y, [4 3 1 2]);
        dzdw = dzdw(end:-1:1, end:-1:1, :, :);
        dzdw = permute(dzdw, [4 3 1 2]);
        return;
    end
        
	% forward
    switch mode
        case 'builtin'
            % use the built-in differentiation method
            %w = w(end:-1:1, end:-1:1, :, :);
            %y = vl_nnconv(dummy_conv_out, w, b, dx, 'pad', pad, 'stride', stride);
            x = permute(x, [3 4 2 1]);
            w = w(end:-1:1, end:-1:1, :, :);
            y = vl_nnconvt(x, w, b, 'upsample', stride, 'crop', [ceil(size(w,1)/4) floor(size(w,1)/4) ceil(size(w,2)/4) floor(size(w,2)/4)]);
        case 'manual'
            % differentiate manually
            w = w(end:-1:1, end:-1:1, :, :);
            y = single( zeros(2*pad+stride*size(dx,1), 2*pad+stride*size(dx,2), size(w,3), size(dx,4)) );
            rows_f = size(w,1);
            cols_f = size(w,2);
            for im_i = 1:size(y,4)
                for f_i = 1:size(w,4)
                    filter = w(:,:,:,f_i);
                    o_i = 1;
                    for i = 1:stride:size(y,1)-size(filter,1)
                        o_j = 1;
                        for j = 1:stride:size(y,2)-size(filter,2)
                            y(i:i+rows_f-1, j:j+cols_f-1, :, im_i) = y(i:i+rows_f-1, j:j+cols_f-1, :,im_i) + ...
                                                                        filter.*dx(o_i,o_j,f_i,im_i);
                            o_j = o_j + 1;
                        end
                        o_i = o_i + 1;
                    end
                end
            end
            y = y(pad+1:end-pad, pad+1:end-pad, :, :);
        case 'transpose_conv'
            w = permute(w, [1 2 4 3]);
            dx2 = single(zeros(stride*size(dx,1), stride*size(dx,2), size(dx,3), size(dx,4)));
            dx2(1:stride:end, 1:stride:end,:,:) = dx;
            stride = 1;
            y = vl_nnconv(dx2, w, b, 'pad', size(w,1)-pad-1, 'stride', stride);
        otherwise
            error(['Deconv does not have mode "' mode '". The available modes are "builtin", "manual", and "transpose_conv".']);
    end
    
    y = permute(y, [4 3 1 2]);

end