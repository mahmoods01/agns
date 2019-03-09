function y = vl_nnreshape_like_theano(x, new_shape, dzdy)
% if new_shape is <=0 for some dimension, then that dimension should be
% fixed
% follow C's row-wise order like in numpy, as opposed to MATLAB's and
% Fortran's column-wise order

    size_x = [size(x,1) size(x,2) size(x,3) size(x,4)];
    new_shape( new_shape<=0 ) = size_x( new_shape<=0 );
    
%   % Slow version    
%     y = single(zeros(new_shape));
%     j1 = 1; j2 = 1; j3 = 1; j4 = 1;
%     for i1 = 1:size_x(1)
%         for i2 = 1:size_x(2)
%             for i3 = 1:size_x(3)
%                 for i4 = 1:size_x(4)
%                     y(j1, j2, j3, j4) = x(i1,i2,i3,i4);
%                     j4 = j4 + 1;
%                     % update y's indices
%                     if j4>new_shape(4)
%                         j4 = 1;
%                         j3 = j3 + 1;
%                         if j3>new_shape(3)
%                             j3 = 1;
%                             j2 = j2 + 1;
%                             if j2>new_shape(2)
%                                 j2 = 1;
%                                 j1 = j1 + 1;
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end

    % faster alternative
    if nargin<3
        x = permute(x, 4:-1:1);
        y = reshape(x, new_shape(end:-1:1));
        y = permute(y, 4:-1:1);
    else
        y = permute(dzdy, 4:-1:1);
        y = reshape(y, [size(x,4), size(x,3) size(x,2) size(x,1)]);
        y = permute(y, 4:-1:1);
    end

end