addpath(pwd);
addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto")


function res = porcCorrectos2(all_theta, X, y)


  etiquetas = rows(all_theta);
  total = length(y);
  res = zeros(etiquetas, total);
  for i=1:etiquetas
    res(i,:) = h((all_theta(i,:))',X);  
  endfor
  
  v = zeros(total, 1);
  for i=1:total
    [maxV ind] = max(res(:,i));
    v(i) = ind;
  endfor
  
  
  res = (sum( y == v)/total)*100;
  
endfunction


function [all_theta] = oneVsAll(X,y, num_etiquetas, lambda) 

n = columns(X);
initial_theta = zeros(n,1);
all_theta = zeros(num_etiquetas,n);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c=1:num_etiquetas 

  [theta] = fmincg(@(t)(lrCostFunction(t,X,(y==c),lambda)), initial_theta, options);
  all_theta(c,:) = theta'; 
  
endfor 

endfunction



function trainMulticlass(filename, lambda, num_etiquetas)

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
  all_theta = oneVsAll(X,y,num_etiquetas,lambda);
  porc = porcCorrectos2(all_theta, Xval, yval)

endfunction