function net = load_eyeglass_generator( filepath, bn_filepath, ngf )
% loading the generator, as implemented in the DCGAN paper

    if nargin==0
        filepath = 'models/gen.mat';
        bn_filepath = 'models/batchnom_parameters_gen.mat';
        ngf = 20;
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
        gen_batchnorm_params = data.gen_batchnorm_params;
    end
    
    layer = struct( 'type', 'dot', 'weights', {matrices(1)} );
    net.layers{end+1} = layer;
    
    if exist('gen_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(2:3)}, ...
                    'mu', gen_batchnorm_params(1).mu, 'v', gen_batchnorm_params(1).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(2:3)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'relu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'reshape_theano', 'new_shape', [-1, ngf*8, 4, 11] );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'deconv', 'weights', {{matrices{4}, []}}, 'stride', 2, 'pad', 2, 'mode', 'builtin' );
    net.layers{end+1} = layer;
    
    if exist('gen_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(5:6)}, ...
                    'mu', gen_batchnorm_params(2).mu, 'v', gen_batchnorm_params(2).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(5:6)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'relu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'deconv', 'weights', {{matrices{7}, []}}, 'stride', 2, 'pad', 2, 'mode', 'builtin' );
    net.layers{end+1} = layer;
    
    if exist('gen_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(8:9)}, ...
                    'mu', gen_batchnorm_params(3).mu, 'v', gen_batchnorm_params(3).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(8:9)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'relu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'deconv', 'weights', {{matrices{10}, []}}, 'stride', 2, 'pad', 2, 'mode', 'builtin' );
    net.layers{end+1} = layer;
    
    if exist('gen_batchnorm_params', 'var')
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(11:12)}, ...
                    'mu', gen_batchnorm_params(4).mu, 'v', gen_batchnorm_params(4).v );
    else
        layer = struct( 'type', 'bnorm_custom', 'weights', {matrices(11:12)}, ...
                    'mu', [], 'v', [] );
    end
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'relu' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'deconv', 'weights', {{matrices{13}, []}}, 'stride', 2, 'pad', 2, 'mode', 'builtin' );
    net.layers{end+1} = layer;
    
    layer = struct( 'type', 'tanh' );
    net.layers{end+1} = layer;
    
end
