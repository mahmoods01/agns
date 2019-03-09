classdef Linear < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnlinear(inputs{1}, params{1}, params{2}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derF, derB] = vl_nnlinear(inputs{1}, params{1}, params{2},  derOutputs{1}) ;
      derParams = {derF, derB} ;
    end

    function obj = Linear(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
