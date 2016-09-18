addpath(pwd);
addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto")


function res = porcCorrectos(theta, X, y)
  
  m = rows(X);
  v = h(theta, X);
  v = v >= 0.5;
  res = (sum( v == y )/m)*100;
  
endfunction



function [theta] = regresionLogistica(X,y, lambda) 

  n = columns(X);
  initial_theta = zeros(n,1);
  options = optimset('GradObj', 'on', 'MaxIter', 500);

  [theta] = fmincg(@(t)(lrCostFunction(t,X,y,lambda)), initial_theta, options);

endfunction

function trainBinary(filename, lambda) 

  load(filename);
  randidx = randperm(rows(x));
  x = x(randidx,:);
  n = columns(x);
  X = x(:,1:(n-1));
  m = rows(X);
  X = [ ones(m,1) X];
  y = x(:,n);
  
  num_train = floor(m*0.7);

  Xval = X((num_train+1):end,:);
  yval = y((num_train+1):end);
  X = X(1:num_train,:);
  y = y(1:num_train);
  
  theta = regresionLogistica(X,y,lambda);

  porcCorrectos(theta,Xval,yval)

endfunction