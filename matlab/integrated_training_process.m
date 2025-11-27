%% Load Training Data & define class catalog & define input image size
disp('Loading training data...')
% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid
clear();
rng(0);

[XData, YData] = digitTrain4DArrayData;

Indices = randperm(numel(YData));

DataSize = size(Indices, 2);
TrainSize = round(0.7*DataSize);
ValidationSize = round(0.2*DataSize);
TestSize = DataSize - TrainSize - ValidationSize;

[XTrain, YTrain] = deal(XData(:,:,:,Indices(1:TrainSize)), YData(Indices(1:TrainSize)));

[XValidation, YValidation] = deal(XData(:,:,:,Indices(TrainSize:TrainSize+ValidationSize)), YData(Indices(TrainSize:TrainSize+ValidationSize)));

[XTest, YTest] = deal(XData(:,:,:,Indices(TrainSize+ValidationSize+1 : end)), YData(Indices(TrainSize+ValidationSize+1 : end)));


NN_layers = [
    imageInputLayer([28,28,1]),
    fullyConnectedLayer(784),
    reluLayer(),
    fullyConnectedLayer(10),
    softmaxLayer()
];

% visualize the neural network
analyzeNetwork(NN_layers);
%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% solver "sgdm" "rmsprop" "adam"

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%%  Train neural network
% define "trainingOptions"
% training using "trainnet"
options = trainingOptions("adam",Plots="training-progress", ExecutionEnvironment="auto");
options.MaxEpochs = 15;
options.InitialLearnRate = 0.01;
options.MiniBatchSize = 100;
options.Metrics = "accuracy";
options.ValidationData = {XValidation, YValidation};

lossFcn = "crossentropy";

net = trainnet(XTrain, YTrain, NN_layers, lossFcn, options);