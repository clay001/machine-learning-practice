function [output] = sigmoid(x)
  output = 1.0 ./ (ones(size(x)) + exp(-x));
end
