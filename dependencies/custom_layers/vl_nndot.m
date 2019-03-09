function [Y, dzdw] = vl_nndot(X, W, dzdy)

    if nargin<3
        Y = X*W;
        dzdw = [];
    else
        Y = dzdy*W';
        dzdw = X'*dzdy;
    end

end