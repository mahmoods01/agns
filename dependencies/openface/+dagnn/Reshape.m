classdef Reshape < dagnn.Layer
  properties
    dims = [1 1 1]
  end

  methods
    function outputs = forward(obj, inputs, ~)
      outputs{1} = vl_nnreshape(inputs{1}, obj.dims);
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      derInputs{1} = vl_nnreshape(inputs{1}, obj.dims, derOutputs{1}) ;
      derParams = {};
    end

    function obj = Reshape(varargin)
        obj.load(varargin{:}) ;
    end
  end
end
