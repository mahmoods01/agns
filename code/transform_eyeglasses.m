function [eyeglass_im, eyeglass_area] = transform_eyeglasses(eyeglass_im, eyeglass_area, tform, out_size)

    logical_area = false;
    if isa(eyeglass_area, 'logical')
        eyeglass_area = uint8(eyeglass_area*255);
        logical_area = true;
    end
    
    if nargin<4
        eyeglass_im = imwarp( eyeglass_im, tform, ...
                                'OutputView', imref2d([size(eyeglass_im,1) size(eyeglass_im,2)]) );
        eyeglass_area = imwarp( eyeglass_area, tform, ...
                                 'OutputView', imref2d([size(eyeglass_area,1) size(eyeglass_area,2)]) );
    else
        eyeglass_im = imwarp(eyeglass_im, tform, 'OutputView', imref2d(out_size));
        eyeglass_area = imwarp(eyeglass_area, tform, 'OutputView', imref2d(out_size));
    end
    
    
    % back to logocal area?
    if logical_area
        eyeglass_area = eyeglass_area>200;
    end
    
end