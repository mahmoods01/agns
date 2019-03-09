function y = vl_nnlrelu(x, dzdy)
% leaky relu
    
    if nargin<2
        y = x;
        y(x<0) = 0.2*x(x<0);
    else
        y = dzdy;
        y(x<0) = 0.2*dzdy(x<0);
    end

end