function y = our_loss(logits, targets, dzdx)
% formalizing the objective of dodging as minimizing
% $logits(t) - sum_{i \neq t} (logits(i))$
% whereas, impersonation, maximizes:
% $logits(t) - sum_{i \neq t} (logits(i))$

    if nargin<3
        % forward pass
        y = zeros(size(logits,4), 1);
        for i = 1:size(logits,4)
            t = targets(i);
            logits_neq_t = gather(logits(:,:,[1:t-1 t+1:end],i));
            y(i) = gather(logits(:,:,t,i)) - sum(logits_neq_t(:));
        end
    else
        % backward pass
        y = ones(size(logits));
        for i = 1:size(logits,4)
            t_i = targets(i);
            y(:,:, t_i, i) = -1;
        end
    end
    
    if isa(logits, 'gpuArray')
        y = gpuArray(y);
    end 

end
