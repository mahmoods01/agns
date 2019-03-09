function y = vl_nnsquare(x, dzdy)
    if nargin<2
        y = x.^2;
    else
        y = (2*x).*dzdy;
    end
end