addpath(pwd);
addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto");


function res = sigmoide(z)

  res = 1 ./ (1 .+ exp(-z));

endfunction

function res = dsigmoide(z)

  res = sigmoide(z).*(1 - sigmoide(z));

endfunction


function res = toVec(num_etiquetas,y)
  res = zeros(num_etiquetas,1);
  res(y) = 1;
endfunction
  
function res = h(Theta1, Theta2, X)
  m = rows(X);
  X = [ ones(m,1) X];

  X = X';
  z1 = sigmoide(Theta1 * X) ; 

  z1 = [ones(1,m); z1 ];

  res = sigmoide(Theta2 * z1);

endfunction

function W= pesosAleatorios (L_in , L_out)
  epsIni = (sqrt(6/(L_in+L_out)));
  W = -epsIni .+ 2*epsIni .* rand(L_out, L_in+1);
  
endfunction

function [ J grad ] = costeRN(params_rn , num_entradas , num_ocultas , num_etiquetas , X, y , lambda )

  m = rows(X);
  Theta1 = reshape (params_rn (1: num_ocultas * ( num_entradas + 1) ) , num_ocultas , ( num_entradas + 1) ) ;
  Theta2 = reshape (params_rn ((1 + (num_ocultas * ( num_entradas + 1) ) ) : end ) , num_etiquetas , ( num_ocultas+ 1) ) ;

  h = h(Theta1,Theta2,X);
 
  J = 0;
  for i = 1:m
    y_aux = toVec(num_etiquetas,y(i));
    for k = 1:num_etiquetas
      J = J +((-y_aux(k)*log(h(k,i))) - ( 1 - y_aux(k))*log(1 - h(k,i)));
    endfor
  endfor
  J = (1/m)*J;

  sumReg = sum((Theta1 .^2)(:)) + sum((Theta2 .^2)(:));
  J = J + (lambda/(2*m))*sumReg;

  grad1 = zeros(rows(Theta1), columns(Theta1));
  grad2 = zeros(rows(Theta2), columns(Theta2));
  for i=1:m
    a1 = X(i,:);
    a1 = [ 1 a1];
    a1 = a1';
    a2 = sigmoide(Theta1 * a1) ; 
    a2 = [ 1; a2 ];
    a3 = sigmoide(Theta2 * a2);
    
    d3 = a3 .- toVec(num_etiquetas,y(i));
    
    d2 = ((Theta2')*d3)(2:end) .* dsigmoide(Theta1 * a1);
    grad1 = grad1 .+ d2*((a1)');
    grad2 = grad2 .+ d3*((a2)');
  endfor
  

  
  
  grad = (1/m) * [grad1(:); grad2(:) ];
  
  

endfunction


function redesNeuronalesProy(filename, lambda, num_ocultas, num_etiquetas, max_iters)
  
  load(filename);
  randidx = randperm(rows(x));
  x = x(randidx,:);
  n = columns(x);
  X = x(:,1:(n-1));
  y = x(:,n);
  
  m = rows(X);

  num_train = floor(m*0.7);

  Xval = X((num_train+1):end,:);
  yval = y((num_train+1):end);
  X = X(1:num_train,:);
  y = y(1:num_train);
  
  checkNNGradients(lambda)
  
  m = rows(X);
  num_entradas = columns(X);
  % 25 ocultas en la practica 4
  % 4 etiquetas
  % 50 - 100 iteraciones
  
  initial_theta = [pesosAleatorios(num_entradas,num_ocultas)(:); pesosAleatorios(num_ocultas, num_etiquetas)(:)];

  options = optimset( 'GradObj' , 'on' , 'MaxIter' , max_iters);

  [theta] = fmincg(@( t ) ( costeRN ( t , num_entradas, num_ocultas, num_etiquetas, X, y, lambda ) ) , initial_theta , options ) ;  
  
  theta1 = reshape (theta (1: num_ocultas * ( num_entradas + 1) ) , num_ocultas , ( num_entradas + 1) ) ;
  theta2 = reshape (theta ((1 + (num_ocultas * ( num_entradas + 1) ) ) : end ) , num_etiquetas , ( num_ocultas+ 1) ) ;
  
  % Calculo porcentaje aciertos
  m = rows(Xval);
  v = zeros(m,1);
  h = h(theta1,theta2,Xval);
  for i=1:m
    [maxV ind] = max(h(:,i));
     v(i) = ind;
  endfor

  porc = (sum( v == yval )/m)*100
  
endfunction
