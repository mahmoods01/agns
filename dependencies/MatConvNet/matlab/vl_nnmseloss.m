function res = vl_nnmseloss(X,Y,dzdy)
% Mean squared error loss
%
% author: Mahmood Sharif


    n = size(X,4);
    
    if isa(X, 'gpuArray')
        Y = gpuArray(Y);
    end

    if nargin <= 2
        % Sum of squared errors
        errors = (X - Y).^2;
        res = zeros(n,1); if isa(X, 'gpuArray'); res = gpuArray(res); end
        
        for i = 1:n
            diff_i = errors(:,:,:,i);
            res(i) = sum(diff_i(:));
        end
        res = sqrt(res);
    else
        % Derivetive dLoss/dX
        % Where Loss = (0.5/n)*sum_{i=1}^{n} [(X_i-Y_i)'(X_i-Y_i)]
        diff = (X-Y);
        res = diff / 2*n;
    end

end