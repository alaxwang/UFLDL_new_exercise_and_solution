%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear all;
addpath(genpath('../common/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
params.m=10000; % num patches
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 64; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity

% Load MNIST data set
data = loadMNISTImages('../common/train-images-idx3-ubyte');

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
display_network(patches(:,1:100));
printf('Orignal patches pause.\n');
%pause;
% Stpatchesep 2) Apply ZCA
org_patches = patches;
[patches, V] = zca2(patches);
display_network(patches(:,1:100));
printf('whined patches pause.\n');
%pause;

% Step 3) Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;
options.display = 'off';
%options.outputFcn = @showBases;

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);

% Check gradient
DEBUG = false 
if DEBUG
   printf('Start softICACost\n');
   fflush(stdout);
   [cost, grad] = softICACost(randTheta, x, params);
   fprintf('End softICACost with grad length = %d\n', length(grad));
   fflush(stdout);
   ngrad = computeNumericalGradient(@(theta) softICACost(theta, x, params),randTheta);
   printf('End compute\n');
   fflush(stdout);
   disp([grad(1:100), ngrad(1:100)]);
   diff = norm(ngrad(1:100)-grad(1:100))/norm(ngrad(1:100)+grad(1:100));
   disp(diff);
   return;
end;

% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); % Use x or xw

% display result
W = reshape(opttheta, params.numFeatures, params.n);
display(size(W));
display(size(x));
display(size(patches));
display_network(W');
printf('final pause.\n');
pause;
%rica_p = W*V; %reshape(W*V, params.numFeatures, params.patchWidth, params.patchWidth);
%rica_p = permute(rica_p, [2,3,1]);
%display(size(rica_p));
%printf('rica pause.\n');
%pause;
%display_network(rica_p*org_patches(:, 1:100));
