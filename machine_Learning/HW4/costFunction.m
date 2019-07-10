function [J,grad,h] = costFunction(theta,x,y)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));

  onevec = ones(size(y));
  J = (-1)./m *sum(y.*log(sigmoid(x*theta)) + (onevec - y).*log(1-sigmoid(x*theta)));

  grad = 1./m *x'*(sigmoid(x*theta)-y);
  
  h = 1./m *x' * diag(sigmoid(x*theta)) * diag(onevec-sigmoid(x*theta)) * x;
  
end

