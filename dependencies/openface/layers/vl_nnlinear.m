function [Y, df, db] = vl_nnlinear(X, W, b, dzdy)

    if nargin<4
        Y = {};
        for i=1:size(X, 4)
            out = mtimes(X(:, :, :, i), W') + b';
            Y{i} = out;
        end
        Y = cat(4, Y{:});
    else
        Y = {};
        df = zeros(size(dzdy, 2), size(X, 2));
        db = zeros(size(dzdy(:, :, :, 1)'));
        for i=1:size(dzdy, 4)
            out = mtimes(dzdy(:, :, :, i), W);
            df = df + mtimes(dzdy(:, :, :, i)', X(:, :, :, i));
            db = db + dzdy(:, :, :, i)';
            Y{i} = out;
        end
        Y = cat(4, Y{:});
        df = df./size(X, 4);
        db = db./size(X, 4);
    end
end