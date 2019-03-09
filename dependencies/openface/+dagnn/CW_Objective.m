classdef CW_Objective < dagnn.Layer
% The loss function used by Carlini and Wagner in the Oakland '17 paper
    
  properties
    dims = [1 1 1]
  end

  methods
    function outputs = forward(~, inputs, ~)
      outputs{1} = carlini_wagner_loss(inputs{1}, inputs{2});
    end

    function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
      derInputs{1} = carlini_wagner_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      derInputs{2} = [] ;
      derParams = {};
    end

    function obj = CW_Objective(varargin)
        obj.load(varargin{:}) ;
    end
  end
end
