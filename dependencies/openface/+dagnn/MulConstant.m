classdef MulConstant < dagnn.ElementWise
%Initialize:
%    layer = dagnn.MulConstant();
%    net.addLayer('layer', layer, {'input'}, {'output'}, {'constant'});
%    index = net.getParamIndex({'constant'});
%    net.params(index).value = c; --c is the constant

  properties
    constant = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnmulconst(inputs{1}, obj.constant) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnmulconst(inputs{1}, obj.constant, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = MulConstant(varargin)
        obj.load(varargin{:});
    end
  end
end
