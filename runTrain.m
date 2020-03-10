clear
commands = categorical(["21","22","23","24"]);
segmentDuration = 400/256;
frameDuration = 0.1;
hopDuration = 0.01;
numBands = 10;
epsil = 1e-6;

load('DS.mat', 'XTrain', 'XTest', 'XValidation', 'YTrain','YTest','YValidation');


sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 4];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'RandXScale',[0.8 1.2], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',augmenter);

classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

timePoolSize = ceil(imageSize(2)/8);
dropoutProb = 0.1;
numF = 25;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(1,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %convolution2dLayer(3,4*numF,'Padding','same')
    %batchNormalizationLayer
    %reluLayer
     
    averagePooling2dLayer([1 timePoolSize])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

miniBatchSize = 30;
validationFrequency = floor(numel(YTrain)/miniBatchSize)*2;
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    ...'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',40);


doTraining = true;
if doTraining
    trainedNet = trainNetwork(augimdsTrain,layers,options);
    save('commandNet.mat','trainedNet');
else
    load('commandNet.mat','trainedNet');
end

YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

YTestPred = classify(trainedNet,XTest);
testError = mean(YTestPred ~= YTest);
disp("Test error: " + testError*100 + "%")







