
function [J, grad] = lrCostFunction(theta,X,y,lambda)
  
  m = length(y);
  n = columns(X);
  
  J = (1/m)*sum(( -y .* log(h(theta,X)) ) .- ( ( 1 .- y) .* (log(1 .- h(theta,X))) )) + (lambda/(2*m))*sum(theta(2:end) .^2); 
  
  grad = (1/m)*X'*(h(theta,X) .- y);
  
  grad(2:n) = grad(2:n) .+ (lambda/m)*theta(2:n);
  
  #for j = 2:n
   # grad(j) = grad(j) + (lambda/m)*theta(j);
  #endfor
