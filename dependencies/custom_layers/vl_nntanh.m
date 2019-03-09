function y = vl_nntanh( x, dzdy )

    if nargin<2
        y = tanh(x);
    else
        y = (1./(cosh(x).^2)).*dzdy;
    end
    
end