%% Load Training Data & define class catalog & define input image size
clear();
rng(0);
disp('Loading training data...')

[XData, YData] = digitTrain4DArrayData;

Indices = randperm(numel(YData));

DataSize = size(Indices, 2);
TrainSize = round(0.7*DataSize);
ValidationSize = round(0.2*DataSize);
TestSize = DataSize - TrainSize - ValidationSize;

[XTrain, YTrain] = deal(XData(:,:,:,Indices(1:TrainSize)), YData(Indices(1:TrainSize)));

[XValidation, YValidation] = deal(XData(:,:,:,Indices(TrainSize+1:TrainSize+ValidationSize)), YData(Indices(TrainSize+1:TrainSize+ValidationSize)));

[XTest, YTest] = deal(XData(:,:,:,Indices(TrainSize+ValidationSize+1 : end)), YData(Indices(TrainSize+ValidationSize+1 : end)));


classes = categories(YTrain);

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%% Define network (dlnet)
NN_layers = [
    imageInputLayer([28,28,1]),
    fullyConnectedLayer(784),
    reluLayer(),
    fullyConnectedLayer(10),
    softmaxLayer()
];
% Create a dlnetwork object from the layer array.
dlnet = dlnetwork(NN_layers);

%% Run task


% ----------------------------------------- Select task -----------------------
task = 1;




% ----------------------- Task 1 ----------------------------------------------
if task == 1 || task == 2
    % Hyperparameters:
    % numEpochs=15 (default)
    % miniBatchSize=100 (default)
    % learnRate=0.01 (default)

    disp("Configuring solver...");
    options = struct();
    options.validate = 1;
    options.showTrainingProgress = 1;
    options.showAccuracy = 1;
    optarg = namedargs2cell(options);
    [solverHandle, initialState] = configureSolver("adam",options);
    disp("Training network...");
    [dlnet] = trainNetwork(dlnet, XTrain, YTrain, XValidation, YValidation, solverHandle, initialState, optarg{:});

    % ----------------------- Task 2 ----------------------------------------------
    if task == 2
        % Hyperparameters:
        % numEpochs=15 (default)
        % miniBatchSize=100 (default)
        % learnRate=0.01 (default)

        % Calculate accuracy on full dataset for displaying average
        accuracyOnFullTestData = calcAccuracy(predictClasses(dlnet, dlarray(XTest, 'SSCB'), classes), YTest);
        
        % Calculate accuracy by digit for each digit 0-9
        accuracyByDigit = [];
        for i=1:10
            accuracyByDigit(i) = calcAccuracyByCategory(dlnet, dlarray(XTest, 'SSCB'), YTest, classes, classes(i));
        end

        % Visualize accuracy data as bar graph
        figure
            hBar = bar(0:9, accuracyByDigit);
    
            % Highlight digits of highest accuracy
            maxIndices = find(accuracyByDigit == max(accuracyByDigit));
    
            % Set the highest bars to have the color green
            highlightColor = [0.43, 0.76, 0.46];
            
            % Apply the color change to the highest bars
            CData = hBar.CData;
            CData(maxIndices, :) = repmat(highlightColor, numel(maxIndices), 1);
            hBar.CData = CData;
            % Ensure CData is used
            hBar.FaceColor = "flat";
    
            % Settings for graph and legend
            ylim([min(accuracyByDigit)-5, 100]);
            hLine = yline(accuracyOnFullTestData, Color="#00A5FF", LineWidth=1.5);
            title("Task 2");
            xlabel("Digit");
            ylabel("Accuracy in %");
            % Dummy bar to be able to show the color in the legend
            hold on;
            hGreenDummy = bar(NaN, NaN, "FaceColor", highlightColor);
            hold off;

            legend([hBar,hGreenDummy,hLine], "Digit Accuracy", "Average On Full Test Data", "Highest Accuracy", Location="southeast");
    end
% ----------------------- Task 3 ----------------------------------------------
elseif task == 3
    % Hyperparameters:
    % numEpochs=15 (default)
    % miniBatchSize=100 (default)
    % learnRate=1e-6:1e-1 (variable)

    % Learn rates that shall be used
    learnRates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    % Available solvers
    solvers = ["adam", "sgdm"];
    % Calculate helper variables for calculation
    nLr = numel(learnRates);
    nSol = numel(solvers);
    n = nLr * nSol;
    % copies is an array containing n copies of dlnet to ensure that the
    % exact same network is trained to ensure comparability
    copies = {};
    % accuracies saves the accuracy value for each learning rate and solver
    accuracies = zeros(n,1);

    fprintf("Amount of trainings: %d\n\n", n);
    % Loop through all cases
    for i=1:n
        % Calculate index for learnRates and solvers array
        lrIndex = mod(i-1, nLr) + 1;
        solIndex = floor((i-1)/nLr) + 1;
        
        fprintf("Training Cycle %d (Solver: '%s', learnRate: %f)\n", i, solvers(solIndex), learnRates(lrIndex));
        disp("Configuring...");

        % Copy dlnet into copies
        copies{i} = dlnetwork(NN_layers);
        %copies{i}.Learnables = dlnet.Learnables;

        % Solver configuration
        options = struct();
        options.learnRate = learnRates(lrIndex);
        optarg = namedargs2cell(options);
        
        [solverHandle, initialState] = configureSolver(solvers(solIndex),options);

        % Train
        disp("Training network...");
        [copies{i}] = trainNetwork(copies{i}, XTrain, YTrain, XValidation, YValidation, solverHandle, initialState, optarg{:});
        
        % Calculate accuracy
        accuracies(i) = calcAccuracy(predictClasses(copies{i}, dlarray(XTest, 'SSCB'), classes), YTest);
    end

    % Visualize accuracy as a function of the learnRate (log-scale on
    % x-axis)
    figure
    for i=0:(nSol-1)
        subplot(2,1,i+1);
        grid on;
        reducedAccuracies = accuracies((i*nLr+1):((i+1)*nLr));
        semilogx(learnRates, reducedAccuracies, 'r-o', 'LineWidth', 1.5);
        ylim([min(reducedAccuracies)-5, 100]);
        title(upper(solvers(i+1)));
        xlabel("Learn Rate");
        ylabel("Accuracy in %");
        legend("Accuracy", Location="southeast");
    end
% ----------------------- Task 4 ----------------------------------------------
elseif task == 4
    % Hyperparameters:
    % numEpochs=15 (default)
    % miniBatchSize=16:256 in 5 steps -> 6 different values (variable)
    % learnRate=optLearnRate=0.01 (default)
    
    % Define variables for calculation
    solver = "adam";
    optLearnRate = 0.01;
    start = 16;
    stop = 256;
    steps = 5;
    batchSizes = start:floor((stop-start)/steps):stop;
    nBatch = numel(batchSizes);

    % copies and accuracies serve the same purpose as in task 3
    copies = {};
    accuracies = zeros(nBatch, 1);
    % trainTimes holds the different times training took for each batch
    % size
    trainTimes = zeros(nBatch, 1);

    % Loop through batchSizes to train with each size once
    for i=1:nBatch
        fprintf("Training Cycle %d (Solver: '%s', learnRate: %f, batchSize: %d)\n", i, solver, optLearnRate, batchSizes(i));
        disp("Configuring...");

        copies{i} = dlnetwork(NN_layers);
        copies{i}.Learnables = dlnet.Learnables;

        % Solver configuration
        options = struct();
        options.miniBatchSize = batchSizes(i);
        options.learnRate = optLearnRate;
        optarg = namedargs2cell(options);

        [solverHandle, initialState] = configureSolver(solver,options);

        % Train
        disp("Training network...");
        tic;
        [copies{i}] = trainNetwork(copies{i}, XTrain, YTrain, XValidation, YValidation, solverHandle, initialState, optarg{:});
        trainTimes(i) = toc;

        % Calculate accuracy
        accuracies(i) = calcAccuracy(predictClasses(copies{i}, dlarray(XTest, 'SSCB'), classes), YTest);
    end

    figure
    subplot(2,1,1);
        plot(batchSizes, accuracies, 'r-o', 'LineWidth', 1.5);
        ylim([min(accuracies)-5, 100]);
        xlabel("Batch Size");
        ylabel("Accuracy in %");
        grid on;
        legend("Accuracy", Location="southeast");
    subplot(2,1,2);
        plot(batchSizes, trainTimes, 'r-o', 'LineWidth', 1.5, 'Color', '#0000FF');
        ylim([min(trainTimes)-5, max(trainTimes)+5]);
        xlabel("Batch Size");
        ylabel("Train Time in s");
        grid on;
        legend("Train Time", Location="southeast");
end

%% Custom function for training
% using a custom solver and options as well as automatic visualization and validation
function [dlnet] = trainNetwork(dlnet, XTrain, YTrain, XValidation, YValidation, solverHandle, solverState, options)
    % Initialize parameters
    arguments
        % Require network
        dlnet
        % Require datasets
        XTrain
        YTrain
        XValidation
        YValidation
        % Require solver
        solverHandle
        solverState
        % Specify Default Training Options (hyperparameters)
        options.miniBatchSize = 100;
        options.numEpochs = 15;
        options.learnRate = 0.01;
        options.validationsPerEpoch = 4;

        options.validate = 0;
        options.showTrainingProgress = 0;
        options.showAccuracy = 0;
    end
    % numIterationsPerEpoch
    options.numIterationsPerEpoch = floor(numel(YTrain)/options.miniBatchSize);
    % The amount of iterations after which the neural network will be evaluated
    % on the validation dataset
    options.validateAfterIterations = floor(options.numIterationsPerEpoch/options.validationsPerEpoch);
    % Available classes of classification problem
    classes = categories(YTrain);
    % Initialize plots for training progress
    % -------------------------------------------------------------------------------
    % TODO: from plots to data export
    
    BatchLoss = [];
    ValidationLoss = [];
    BatchAccuracy = [];
    ValidationAccuracy = [];

    if options.showTrainingProgress
        figure
        subplot(2,1,1)
            yyaxis left
            BatchLoss = animatedline(0, 0, Color="#0000FF");
            yline([0, inf])
            xlabel("Iteration")
            ylabel("Loss")
            grid on
            yyaxis left
            ValidationLoss = animatedline(0, 0, Color="#00A5FF", LineWidth=1.5);
            yline([0, inf])
            xlabel("Iteration")
            ylabel("Loss")
            grid on
            yyaxis right
            BatchAccuracy = animatedline(0, 0, Color="#FF0000");
            yline([0, inf])
            xlabel("Iteration")
            ylabel("Accuracy")
            grid on
            yyaxis right
            ValidationAccuracy = animatedline(0, 0, Color="#FFA500", LineWidth=1.5);
            yline([0, inf])
            xlabel("Iteration")
            ylabel("Accuracy")
            grid on
    end

    % Training loop
    for epoch = 1:options.numEpochs
        
        % Random permutation of training data
        idx = randperm(numel(YTrain));
        for i = 1:options.numIterationsPerEpoch
            % Calculate iteration
            iteration = (epoch-1)*options.numIterationsPerEpoch + i;

            % Validation accuracy and loss (calculation and visualization)
            if options.validate
                % Check if a validation cycle is due for the current iteration
                if mod(iteration-1, options.validateAfterIterations) == 0
                    % Evaluate modelGradients on validation data
                    [~, loss, dlYPred] = dlfeval(@modelGradients, dlnet, dlarray(single(XValidation), 'SSCB'), single(transpose(onehotencode(YValidation, 2))));
                    
                    if options.showTrainingProgress
                        % Visualize validation loss
                        addpoints(ValidationLoss, iteration-1, loss);
                        % Calculate and visualize accuracy on validation data
                        if options.showAccuracy
                            validAccuracy = calcAccuracy(predictionToClasses(dlYPred, classes), YValidation);
                            addpoints(ValidationAccuracy, iteration-1, validAccuracy);
                        end
                        drawnow
                    end
                end
            end
    
            % Read mini-batch of data and convert the labels to dummy variables.
            indicesMB = idx((options.miniBatchSize*(i-1)+1):options.miniBatchSize*i);
            featureMB = XTrain(:,:,:,indicesMB);
            labelMB = YTrain(indicesMB);
    
            % Convert mini-batch of data to a dlarray.
            dlX = dlarray(single(featureMB), 'SSCB');
            Y = transpose(onehotencode(labelMB, 2));
            Y = single(Y);
            
            % Evaluate the model gradients and loss using dlfeval and the
            % modelGradients helper function.
            [grad, loss, dlYPred] = dlfeval(@modelGradients, dlnet, dlX, Y);
            
            % Update the network parameters using the user defined
            % optimizer
            [dlnet,solverState] = ...
                solverHandle(dlnet,grad,solverState,options.learnRate);
            
    
            
            % Calculate accuracy & show the training progress.
            if options.showTrainingProgress
        
                %---add new values to plot
                addpoints(BatchLoss, iteration, loss);
                if options.showAccuracy
                    accuracy = calcAccuracy(predictionToClasses(dlYPred, classes), labelMB);
                    addpoints(BatchAccuracy, iteration, accuracy);
                end
                drawnow
            end
        end
    end
end

%% Model Gradients Function
function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y)

    % forward propagation 
    dlYPred = forward(dlnet,dlX);
    % calculate loss -- varies based on different requirement
    loss = crossentropy(dlYPred,Y);
    % calculate gradients 
    gradients = dlgradient(loss,dlnet.Learnables);
end

%% Utility functions
% Convert a prediction (size 10x?) to an array of predicted classes (size ?)
function [predClasses] = predictionToClasses(dlYPred, classes)
    % Get array of indices of predicted classes (equivalent to indices of max
    % values)
    [~, idxPred] = max(dlYPred, [], 1);
    % Convert array of indices to array of predicted classes and return
    predClasses = classes(idxPred);
end

% Get predictions of network on dataset with given classes
function [predClasses] = predictClasses(dlnet, XData, classes)
    % Feed data forward through the network and convert using
    % predictionToClasses(...)
    dlYPred = forward(dlnet, XData);
    predClasses = predictionToClasses(dlYPred, classes);
end

% Calculate accuracy from predicted classes and corresponding labels
function [avgAccuracy] = calcAccuracy(predClasses, YData)
    % Calculate amount of correctly classified examples
    correct = sum(predClasses == reshape(YData, size(predClasses)));
    avgAccuracy = correct/numel(YData)*100;
end

% Calculate accuracy by category
function [avgAccuracy] = calcAccuracyByCategory(dlnet, XData, YData, classes, category)
    % Reduce data to only include examples of the desired category
    reducedIndices = find(YData==category);
    reducedYData = YData(reducedIndices);
    reducedXData = XData(:,:,:,reducedIndices);
    % Predict classes and calculate accuracy
    predClasses = predictClasses(dlnet, reducedXData, classes);
    avgAccuracy = calcAccuracy(predClasses, reducedYData);
end



%% Solver selection and wrappers

% This function returns the function handle for the selected solver as well
% as an initial state containing important state variables custom to the
% respective solver algorithm
function [solverHandle, initialState] = configureSolver(solverName, options)
    % Initialize initial state
    initialState = struct();

    switch lower(solverName)
        case "adam"
            % Populating initial state for adam solver
            initialState.averageGrad = [];
            initialState.averageSqGrad = [];
            initialState.iteration = 1;
            % Creating function handle for adam solver
            solverHandle = @(dlnet, grad, state, learnRate) adamWrapper(dlnet, grad, state, setfield(options, 'learnRate', learnRate));
        case "sgdm"
            % Populating initial state for sgdm solver
            initialState.vel = [];
            % Creating function handle for sgdm solver
            solverHandle = @(dlnet, grad, state, learnRate) sgdmWrapper(dlnet, grad, state, setfield(options, 'learnRate', learnRate));
        otherwise
            error("Unknown solver '" + solverName + "'.");
    end
end

% Wrapper for sgdmupdate
function [dlnet, state] = sgdmWrapper(dlnet, grad, state, options)
    % Order fields for argument list
    [args] = orderFields(options, {'momentum'});
    % Compute sgdmupdate (options.learnRate is written explicitly to ensure
    % it has been defined).
    [dlnet, state.vel] = sgdmupdate(dlnet,grad,state.vel,options.learnRate, args{:});
end

% Wrapper for adamupdate
function [dlnet, state] = adamWrapper(dlnet, grad, state, options)
    % Order fields for argument list
    [args] = orderFields(options, {'gradDecay', 'gradSqDecay', 'epsilon'});
    % Compute adamupdate (options.learnRate is written explicitly to ensure
    % it has been defined -- as in sgdmWrapper).
    [dlnet, state.averageGrad, state.averageSqGrad] = adamupdate(dlnet, grad, state.averageGrad, state.averageSqGrad, state.iteration, options.learnRate, args{:});
    % Increase iteration counter as adamupdate requires the current
    % iteration as one of its state variables
    state.iteration = state.iteration +1;
end

% Orders a struct of fields according to a given order
function [fields] = orderFields(options, fieldOrder)
    % Ordered fields
    fields = {};
    % Loop through fieldOrder
    for i=1:numel(fieldOrder)
        % Populate fields with the field from options that should be next
        % according to the field order, if it exists.
        % Otherwise: stop as order matters in argument lists.
        if isfield(options, fieldOrder(i))
            fields{i} = options.(fieldOrder(i));
        else
            break;
        end
    end
end