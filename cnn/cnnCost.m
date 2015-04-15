function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

activations = activations + cnnConvolve(filterDim, numFilters, images, Wc, bc);
% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = activationsPooled + cnnPool(poolDim, activations); % outputDim*outputDim*num_filter*num_images 

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);
z = exp(bsxfun(@plus, Wd*activationsPooled, bd)); 
%% numClass * hidden * hidden *num_image = numclass * num_image 
%z = exp(bsxfun(@minus,z,max(z,[],1)));
probs = probs + bsxfun(@rdivide, z, sum(z, 1));
lambda = 0.0001;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

cost = 0;
lables_full = full(sparse(labels, 1:numImages, 1)); % num_class * num_image 
cost = cost - 1/numImages * sum(sum(log(probs) .* lables_full));
cost = cost + (lambda/2) * (sum(Wc(:).^2) + sum(Wd(:).^2));


softmax_error = probs - lables_full; % num_class * num_images
pool_error = Wd' * softmax_error; % hidden * num_images
pool_error = reshape(pool_error, [], outputDim, numFilters, numImages);
mx_one = ones(poolDim, poolDim);
upool_error = zeros(convDim, convDim, numFilters, numImages);
for image=1:numImages
  for filter=1:numFilters
    upool_error(:, :, filter, image) = ...
       kron(pool_error(:, :, filter, image), mx_one) / (poolDim^2);
  end;
end;

%printf('before detal\n');
%fflush(stdout);

conv_error = upool_error .* activations .* (1 - activations); 
% convDim,convDim, numFilters, numImages


Wd_grad = Wd_grad + (1/numImages) * softmax_error * activationsPooled';  + lambda .* Wd;
bd_grad = bd_grad + (1/numImages) * sum(softmax_error, 2);

%printf('before detal-1\n');
%fflush(stdout);
for filter=1:numFilters
  for image=1:numImages
    Wc_grad(:, :, filter) = Wc_grad(:, :, filter) + ...
      conv2(images(:, :, image), rot90(conv_error(:, :, filter, image),2), 'valid'); 
  end;
  Wc_grad(:, :, filter) = Wc_grad(:, :, filter) / numImages;
end;

%printf('before detal-2\n');
%fflush(stdout);

for filter=1:numFilters
  bc_grad(filter) = sum((conv_error(:,:,filter,:))(:)) / numImages;


Wc_grad = Wc_grad + lambda * Wc;  
%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
