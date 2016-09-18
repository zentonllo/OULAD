addpath(pwd);
addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto");

function sim = gaussianKernel(x1,x2,sigma)
  sim = exp( -(sum( (x1-x2).^2 ) / (2*(sigma^2))) );

endfunction


function res = porCorrect(vSal,y)
  a = sum( vSal == y );
  b = length(y);
  res = (a/b)*100;
endfunction


function chooseParam(X,y,Xval,yval)
  params = [0.01;0.03;0.1;0.3;1;3;10;30];
  l = length(params);
  bestPorc = 0;
  
  for i= 1:l
    C = params(i);
    for j = 1:l
       sigma = params(j);
       model = svmTrain(X,y,C, @(x1,x2) gaussianKernel(x1,x2,sigma));
       vSal = svmPredict(model,Xval);
       porc = porCorrect(vSal,yval);
       if ( porc >= bestPorc ) 
        bestModel = model;
        bestPorc = porc;
        bestC = C;
        bestSigma = sigma;
       endif
    endfor
    
  endfor
  
  bestPorc
  bestC
  bestSigma
  
  
  
  
endfunction

function SVMProy(filename)

load(filename);
n = columns(x);
randidx = randperm(rows(x));
x = x(randidx,:);
X = x(:,1:(n-1));
y = x(:,n);

m = rows(X);

num_train = floor(m*0.7);

Xval = X((num_train+1):end,:);
yval = y((num_train+1):end);
X = X(1:num_train,:);
y = y(1:num_train);


chooseParam(X,y,Xval,yval)


endfunction