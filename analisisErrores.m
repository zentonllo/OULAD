addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto")
addpath(pwd)

function [err fp fn] = error(theta, X, y)
	
 m = length(y);
 
 y_pred = h(theta,X) >= 0.5;
 err = (1/m)*sum( abs( y_pred - y) );

 total_errores = sum( abs( y_pred - y) );
 balance = sum( y_pred - y );
 fp = (balance + total_errores)/2;
 fn = total_errores - fp;
  
 
endfunction


function err = error2(all_theta, X, y)


  all_theta = all_theta';
  m = length(y);

  res = h(all_theta,X);
  
  y_pred = zeros(m, 1);

  for i=1:m
    [maxV ind] = max(res(i,:));
    y_pred(i) = ind;
  endfor
  
  
  err = (1/m)*(m - sum( y == y_pred ));


endfunction

function M = genMatPol(v, p)
  M = v;
  for i = 2:p
    M = [M (v .^i)];
  endfor
endfunction


function pruebasRegLogBin(filename, lambda, max_iters, lambda_final)
    
    load(filename);
    n = columns(x);
    randidx = randperm(rows(x));
    x = x(randidx,:);
    
    X = x(:,1:(n-1));
    m = rows(X);
    X = [ ones(m,1) X];
    y = x(:,n);
    
    %%%%%%%% Descomentar para añadir valores polinómicos al atributo número convocatorias utilizadas
%    p = 6;
%    X = x(:,1:(n-3));
%    X2 = genMatPol(x(:,(n-2)), p);
%    X = [X X2 x(:,(n-1))];
%    n = columns(X);
    %%%%%%%%%%%%%%%%%%%% 
    
    initial_theta = zeros(n,1);

    num_train = floor(m*0.6);
    num_val = num_train + 1 + floor(m*0.2);


    Xval = X((num_train+1):num_val,:);
    yval = y((num_train+1):num_val);

    Xtest = X((num_val+1):end,:);
    ytest = y((num_val+1):end);

    X = X(1:num_train,:);
    y = y(1:num_train);

    m = rows(X);


    errEnt = zeros(m,1);
    errVal = zeros(m,1);
    fpEnt = zeros(m,1);
    fpVal = zeros(m,1);
    fnEnt = zeros(m,1);
    fnVal = zeros(m,1);

   options = optimset( 'GradObj' , 'on' , 'MaxIter' , max_iters);

   for i = 1:m
   	[theta] = fmincg(@(t)(lrCostFunction(t,X(1:i,:),y(1:i),lambda)), initial_theta, options);
   	[errEnt(i) fpEnt(i) fnEnt(i)] = error(theta, X(1:i,:), y(1:i));
   	[errVal(i) fpVal(i) fnVal(i)] = error(theta, Xval, yval);
   endfor


%Curvas de aprendizaje
    figure(1);
    hold on;
    axis([1,m,0,1]);
    title( "Curvas de aprendizaje para la regresion logistica binaria","fontsize", 12 );
    xlabel("Numero de ejemplos de entrenamiento", "fontsize", 10);
    ylabel("Error (en tanto por uno)","fontsize",10);
    x_axis = (1:1:m)';
    plot(x_axis, errEnt, "color", "red");
    plot(x_axis, errVal, "color", "blue");
    legend("Entrenamiento", "Validacion");
    
%Ploteo evolucion falsos positivos y falsos negativos
    figure(2);
    hold on;
    axis([1,m,0,m]);
    title( "Evolucion del numero de falsos positivos y falsos negativos (Datos de entrenamiento)","fontsize", 12 );
    xlabel("Numero de ejemplos de entrenamiento", "fontsize", 10);
    ylabel("Numero de falsos positivos y negativos","fontsize",10);
    x_axis = (1:1:m)';
    plot(x_axis, fpEnt, "color", "red");
    plot(x_axis, fnEnt, "color", "blue");
    legend("Falsos positivos", "Falsos negativos");
    
    
    figure(3);
    hold on;
    axis([1,m,0,m]);
    title( "Evolucion del numero de falsos positivos y falsos negativos (Datos de validacion)","fontsize", 12 );
    xlabel("Numero de ejemplos de validacion", "fontsize", 10);
    ylabel("Numero de falsos positivos y negativos","fontsize",10);
    x_axis = (1:1:m)';
    plot(x_axis, fpVal, "color", "red");
    plot(x_axis, fnVal, "color", "blue");
    legend("Falsos positivos", "Falsos negativos");

% Elección Lambda


  lambda_vec = [0; 0.001; 0.003; 0.01;  0.03; 0.1; 0.3; 1; 3; 10; 20; 30; 100; 300];
   l = length(lambda_vec);
   
   errEnt = zeros(l,1);
   errVal = zeros(l,1);

   initial_theta = zeros(n,1);

   for i = 1:l
     lambda = lambda_vec(i);
     [theta] = fmincg(@(t)(lrCostFunction(t,X,y,lambda)), initial_theta, options);
     [errEnt(i) dummy dummy2] = error(theta, X, y);
     [errVal(i) dummy dummy2] = error(theta, Xval, yval);
   endfor
   
   figure(4);
   hold on;
   axis([0,lambda_vec(l),0,1]);
   title( "Valores de error para los distintos valores lambda","fontsize", 12 );
   xlabel("Valor de lambda", "fontsize", 10);
   ylabel("Error (en tanto por uno)","fontsize",10);
   plot(lambda_vec, errEnt, "color", "red", "linewidth", 1);
   plot(lambda_vec, errVal, "color", "blue", "linewidth", 1);
   legend("Entrenamiento", "Validacion");
   
   lambda = lambda_final;
   [theta] = fmincg(@(t)(lrCostFunction(t,X,y,lambda)), initial_theta, options);
   [errorFinal fpFinal fnFinal] = error(theta, Xtest, ytest)

    
endfunction 


function pruebasRegLogMult(filename, lambda, max_iters, num_etiquetas, lambda_final)

    load(filename);
    n = columns(x);
    randidx = randperm(rows(x));
    x = x(randidx,:);
    
    X = x(:,1:(n-1));
    m = rows(X);
    X = [ ones(m,1) X];
    y = x(:,n);
    %%%%%%%% Descomentar para añadir valores polinómicos al atributo número convocatorias utilizadas
%    p = 3;
%    X = x(:,1:(n-3));
%    X2 = genMatPol(x(:,(n-2)), p);
%    X = [X X2 x(:,(n-1))];
%    n = columns(X);
    %%%%%%%%%%%%%%%%%%%%

    
    
    initial_theta = zeros(n,1);

    num_train = floor(m*0.6);
    num_val = num_train + 1 + floor(m*0.2);


    Xval = X((num_train+1):num_val,:);
    yval = y((num_train+1):num_val);

    Xtest = X((num_val+1):end,:) ;
    ytest = y((num_val+1):end);

    X = X(1:num_train,:);
    y = y(1:num_train);

    m = rows(X);


    initial_theta = zeros(n,1);

    all_theta = zeros(num_etiquetas,n);
    options = optimset( 'GradObj' , 'on' , 'MaxIter' , max_iters);
    
    errEnt = zeros(m,1);
    errVal = zeros(m,1);

    for i = 1:m
    	for c=1:num_etiquetas 
      		[theta] = fmincg(@(t)(lrCostFunction(t,X(1:i,:),(y(1:i)==c),lambda)), initial_theta, options);
      		all_theta(c,:) = theta'; 
    	endfor 

    	errEnt(i) = error2(all_theta, X(1:i,:), y(1:i));
    	errVal(i) = error2(all_theta, Xval, yval);
    endfor



    %Curvas de aprendizaje
     figure(1);
     hold on;
     axis([1,m,0,1]);
     title( "Curvas de aprendizaje para la regresion logistica multiclase","fontsize", 12 );
     xlabel("Numero de ejemplos de entrenamiento", "fontsize", 10);
     ylabel("Error (en tanto por uno)","fontsize",10);
     x_axis = (1:1:m)';
     plot(x_axis, errEnt, "color", "red");
     plot(x_axis, errVal, "color", "blue");
     legend("Entrenamiento", "Validacion");


   %Elección Lambda

    lambda_vec = [0; 0.001; 0.003; 0.01;  0.03; 0.1; 0.3; 1; 3; 10; 20; 30; 100; 300];
    l = length(lambda_vec);
     
    errEnt = zeros(l,1);
    errVal = zeros(l,1);

    initial_theta = zeros(n,1);


    for i = 1:l
       lambda = lambda_vec(i);
       for c=1:num_etiquetas 
           [theta] = fmincg(@(t)(lrCostFunction(t,X,(y==c),lambda)), initial_theta, options);
           all_theta(c,:) = theta'; 
       endfor 

       errEnt(i) = error2(all_theta, X, y);
       errVal(i) = error2(all_theta, Xval, yval);
     endfor
     
    figure(2);
    hold on;
    axis([0,lambda_vec(l),0,1]);
    title( "Valores de error para los distintos valores lambda","fontsize", 12 );
    xlabel("Valor de lambda", "fontsize", 10);
    ylabel("Error (en tanto por uno)","fontsize",10);
    plot(lambda_vec, errEnt, "color", "red", "linewidth", 1);
    plot(lambda_vec, errVal, "color", "blue", "linewidth", 1);
    legend("Entrenamiento", "Validacion");
     
     lambda = lambda_final;
     for c=1:num_etiquetas 
           [theta] = fmincg(@(t)(lrCostFunction(t,X,(y==c),lambda)), initial_theta, options);
           all_theta(c,:) = theta'; 
       endfor
     errorFinal = error2(all_theta, Xtest, ytest)


endfunction














