function [y,dzdg,dzdb] = vl_nnbnorm_custom(x, g, b, mu, v, dzdy)

    eps = 1e-8;
    
    dzdg = []; dzdb = [];
    
    if nargin<6
        % forward pass
        if isempty(mu) % are moment provided?
            mu = mean(mean(mean(x,4),3),1);
            v = (bsxfun(@minus, x, mu)).^2;
            v = mean(mean(mean(v,4),3),1);
            x = permute(x, [3 4 2 1]);
            y = vl_nnbnorm(x, g, b, 'Moments', [mu', sqrt(v'+eps)]) ;
            y = permute(y, [4 3 1 2]);
        else
            x = permute(x, [3 4 2 1]);
            y = vl_nnbnorm(x, g, b, 'Moments', [mu', sqrt(v'+eps)]) ;
            y = permute(y, [4 3 1 2]);
        end
    else
        % backward pass
        if isempty(mu) % are moment provided?
            mu = mean(mean(mean(x,4),3),1);
            v = (bsxfun(@minus, x, mu)).^2;
            v = mean(mean(mean(v,4),3),1);
            x = permute(x, [3 4 2 1]);
            dzdy = permute(dzdy, [3 4 2 1]);
            [y, dzdg, dzdb] = vl_nnbnorm(x, g, b, dzdy, 'Moments', [mu', sqrt(v'+eps)]) ;
            dzdg = reshape(dzdg, size(g));
            dzdb = reshape(dzdb, size(b));
            y = permute(y, [4 3 1 2]);
        else
            x = permute(x, [3 4 2 1]);
            dzdy = permute(dzdy, [3 4 2 1]);
            [y, dzdg, dzdb] = vl_nnbnorm(x, g, b, dzdy, 'Moments', [mu', sqrt(v'+eps)]) ;
            dzdg = reshape(dzdg, size(g));
            dzdb = reshape(dzdb, size(b));
            y = permute(y, [4 3 1 2]);
        end
    end

end