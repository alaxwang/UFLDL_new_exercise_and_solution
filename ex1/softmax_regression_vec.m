function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta=[theta,zeros(size(theta,1),1)];
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  M = exp(theta'*X);
  y_m = full(sparse(y,1:m,1)); % num_class * m
  I = sub2ind(size(M),y,1:size(M,2));
  f = -sum(log(M(I)./sum(M,1)));
  g = -(X*(y_m - bsxfun(@rdivide,M,sum(M,1)))');
  g(:,end)=[];
  g=g(:); % make gradient a vector for minFunc
end
