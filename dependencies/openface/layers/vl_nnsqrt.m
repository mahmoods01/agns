function y = vl_nnsqrt(x, dzdy)
    if nargin<2
        y = x.^(0.5);
    else
        prod = 0.5*(x.^(-0.5));
        prod(x==0) = 0;
        y = dzdy.*prod;
    end
end