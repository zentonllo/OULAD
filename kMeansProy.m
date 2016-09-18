addpath(pwd);
addpath("B:\\Documentos\\UCM\\Erasmus\\AABD\\proyecto");

function centroids = randomCentroids(X,k)
  m = rows(X);  
  randidx = randperm(m);
  centroids = X(randidx(1:k),:);

endfunction

function clusteringProyec(filename, num_centroids, max_iters)
  

  load(filename);
  initial_centroids = randomCentroids(x,num_centroids);
  plot_progress = false;
  [centroids, idx] = runkMeans(x, initial_centroids, max_iters, plot_progress);

  centroids
endfunction