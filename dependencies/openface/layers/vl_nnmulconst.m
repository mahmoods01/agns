function y = vl_nnmulconst(x, c, dzdy)
    if nargin == 2
        y = x.*c;
    else
        y = dzdy .* c;
    end
end