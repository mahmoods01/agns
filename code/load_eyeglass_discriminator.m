function net = load_eyeglass_discriminator( filepath, bn_filepath )
% loading the discriminator, as implemented in the DCGAN paper

    if nargin==0
        filepath = 'models/discrim.mat';
        bn_filepath = 'models/batchnom_parameters_discrim.mat';
    end

    net = struct('layers', [], 'im_size', [], 'crop', []);
    
    % info about the image size and crop
    net.im_size = [224 224];
    net.crop = [53 25 53+64-1 25+176-1];

    % weights and biases
    data = load(filepath);
    matrices = data.matrices;
    
    % batch norm parameters, if file exists
    if ~isempty( bn_filepath )
        data = load(bn_filepath);
        discrim_batchnorm_params = data.discrim_batchnorm_params;
    end
    
    layer = struct( 'type', 'myconv', 'weights', {{matrices{1}, []}}, 'stride', 2, 'pad', 2 );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'lrelu' );
    net.layers{end+1} = layer;
     
    layer = struct( 'type', 'myconv', 'weights', {{matrices{2}, []}}, 'stride', 2, 'pad', 2 );
    net.layers{end+1} = layer;
    
    if exist('discrim_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(3:4)}, ...
                    'mu', discrim_batchnorm_params(1).mu, 'v', discrim_batchnorm_params(1).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(3:4)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'lrelu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'myconv', 'weights', {{matrices{5}, []}}, 'stride', 2, 'pad', 2 );
    net.layers{end+1} = layer;
    
    if exist('discrim_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(6:7)}, ...
                    'mu', discrim_batchnorm_params(2).mu, 'v', discrim_batchnorm_params(2).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(6:7)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'lrelu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'myconv', 'weights', {{matrices{8}, []}}, 'stride', 2, 'pad', 2 );
    net.layers{end+1} = layer;
    
    if exist('discrim_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(9:10)}, ...
                    'mu', discrim_batchnorm_params(3).mu, 'v', discrim_batchnorm_params(3).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(9:10)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'relu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'reshape_theano', 'new_shape', [-1, numel(matrices{11}), 1, 1] );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'dot', 'weights', {matrices(11)} );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'sigmoid' );
    net.layers{end+1} = layer;
    
end
