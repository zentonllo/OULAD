function centroids = computeCentroids(X, idx, k)
m = rows(X);
n = columns(X);
card = zeros(k,m);
lengths = ones(k,1);
for i = 1:m
  j = idx(i);
  card(j,lengths(j)) = i;
  lengths(j) = lengths(j) + 1; 
endfor

lengths = lengths - ones(k,1);
centroids = zeros(k,n);
for i=1:k 
  
  sum = zeros(1,n);
  l = lengths(i);
  for j = 1:l
    sum = sum + X(card(i,j),:);
  endfor
  
  centroids(i,:) = (1/l)*sum;

endfor

