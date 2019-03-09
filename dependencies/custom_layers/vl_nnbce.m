function y = vl_nnbce(x, p, dzdy)
% binary cross-entropy

    eps = 1e-8;

    if nargin<3
        % forward
        y = -p.*log(x+eps) - (1-p).*log(1-x+eps);
        y = mean(y);
    else
        % backward
        y = -(p - 1)./(eps - x + 1) - p./(eps + x);
        y = y/numel(x);
    end

end