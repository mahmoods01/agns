function [y, dzdw, dzdb] = vl_nnmyconv(x, w, b, pad, stride, dzdy)
% Convlution function that is compatible with Theano's shaping of matrices

    x = permute(x, [3 4 2 1]);
    w = permute(w, [3 4 2 1]);
    w = w(end:-1:1, end:-1:1, :, :);
    dzdw = []; dzdb = [];
    if nargin<6
        y = vl_nnconv(x, w, b, 'pad', pad, 'stride', stride);
    else
        dzdy = permute(dzdy, [3 4 2 1]);
        [y, dzdw, dzdb] = vl_nnconv(x, w, b, dzdy, 'pad', pad, 'stride', stride);
        dzdw = dzdw(end:-1:1, end:-1:1, :, :);
        dzdw = permute(dzdw, [4 3 1 2]);
    end
    y = permute(y, [4 3 1 2]);

end