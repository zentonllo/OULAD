function idx = findClosestCentroids(X,centroids)
  m = rows(X);
  k = rows(centroids);
  
  idx = zeros(m,1);

  for i = 1:m
    min = norm(X(i,:) - centroids(1,:))^2;
    idx(i) = 1;
    for j = 2:k
      val = norm(X(i,:) - centroids(j,:) )^2;
      if( val < min ) 
        min = val;
        idx(i) = j;
      endif
    endfor
  endfor

