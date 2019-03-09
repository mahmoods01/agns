classdef Our_Objective < dagnn.Layer
% The loss function used by us in the NDSS '17 paper
    
  properties
    dims = [1 1 1]
  end

  methods
    function outputs = forward(~, inputs, ~)
      outputs{1} = our_loss(inputs{1}, inputs{2});
    end

    function [derInputs, derParams] = backward(~, inputs, ~, derOutputs)
      derInputs{1} = our_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      derInputs{2} = [] ;
      derParams = {};
    end

    function obj = Our_Objective(varargin)
        obj.load(varargin{:}) ;
    end
  end
end
