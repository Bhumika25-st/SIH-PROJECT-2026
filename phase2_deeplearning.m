%% Phase 2: DR Severity Grading (CPU Optimized v7 - corrected)
% RETINA-AEGIS Module 2
%
% FIXES APPLIED vs the version you were running:
%   1. CRITICAL: double-normalization bug. cpuRead() manually applies
%      ImageNet mean/std normalization, but resnet50/resnet18's default
%      imageInputLayer ALSO applies its own 'zerocenter' normalization
%      (subtracting a learned average-image expecting raw 0-255 pixels).
%      Every image was being normalized twice, badly distorting the
%      input the network actually sees. The input layer is now replaced
%      with 'Normalization','none' since cpuRead already handles it.
%   2. Saved model renamed to RetinaAegis_Classifier.mat so Phase 3
%      (Grad-CAM) and Phase 3B (calibration/uncertainty) can load it
%      without path changes. (Previously RetinaAegis_CPU.mat.)
%   3. Script-relative paths (scriptDir), matching Phase 1/Phase 3, so
%      this runs correctly regardless of MATLAB's current folder.
%   4. Saves the exact Val/Test split (files + labels) to
%      Phase2_ValidationSplit.mat for Phase 3B. Use imdsVal for
%      calibration/threshold selection (imdsTest should stay fully
%      untouched for final unbiased reporting, since ValidationPatience
%      already uses imdsVal during training - some minor leakage there
%      is normal and accepted, but doubling up on imdsTest would not be).

clear; clc; close all; fclose all;

%% 1. CONFIGURATION
scriptDir = fileparts(mfilename('fullpath'));

processedDir = fullfile(scriptDir, 'aptos2019-blindness-detection', 'processed_v5', 'gradeable');
csvFile      = fullfile(scriptDir, 'aptos2019-blindness-detection', 'train.csv');

if ~exist(processedDir, 'dir')
    error(['Run Module 1 first. Missing: %s\n' ...
           'Phase 1 v5 must reach MODE=''infer'' or MODE=''quick_sort'' ' ...
           '(MODE=''extract'' alone does not write image files).'], processedDir);
end
if ~exist(csvFile, 'file')
    error('Label file not found: %s', csvFile);
end

nThreads = maxNumCompThreads;
fprintf('CPU threads available: %d\n\n', nThreads);

% ---- NETWORK CHOICE ----
% 'resnet50' -> heavier, slightly higher potential accuracy, slow on CPU
% 'resnet18' -> ~3-4x faster to train on CPU, small accuracy trade-off
% For CPU-only training, resnet18 is the recommended default.
CFG.network = 'resnet50';

% ---- FREEZE EARLY LAYERS ----
% Only fine-tune the last N layers + final FC. This is the single
% biggest CPU speed lever: gradients are not computed for frozen
% layers, cutting most of the backward-pass compute.
CFG.freezeUpToPercent = 0.75;  % freeze first 75% of layers

CFG.inputSize   = [224 224 3];
CFG.batchSize   = 32;       % smaller batch = safer on CPU-only memory/compute
CFG.maxEpochs   = 20;
CFG.initialLR   = 1e-3;
CFG.valFreq     = 50;
CFG.valPatience = 4;
CFG.useTTA      = true;
CFG.numTTA      = 5;
CFG.TTAchunk    = 48;
% Parallel Computing Toolbox NOT required for this script.

fprintf('Network: %s | Batch: %d | Epochs: %d | TTA: %d\n', ...
    CFG.network, CFG.batchSize, CFG.maxEpochs, CFG.numTTA);

%% 2. LOAD DATA
labelTable = readtable(csvFile);

if iscategorical(labelTable.diagnosis)
    labelTable.diagnosis = double(labelTable.diagnosis);
elseif iscell(labelTable.diagnosis)
    labelTable.diagnosis = cellfun(@str2double, labelTable.diagnosis);
end

imds = imageDatastore(processedDir, 'FileExtensions', {'.png','.jpg','.jpeg'});

if isempty(imds.Files)
    error(['%s exists but contains no images. Check the Phase 1 v5 run ' ...
           'summary for how many images were flagged Gradeable.'], processedDir);
end

[~, baseNames, ~] = fileparts(imds.Files);
imgIds = string(baseNames);
csvIds = string(labelTable.id_code);

[isMatched, matchIdx] = ismember(imgIds, csvIds);

if ~any(isMatched)
    error('No image matched any CSV row.');
end

if any(~isMatched)
    fprintf('Dropped: %d images (no labels)\n', sum(~isMatched));
    imds = subset(imds, find(isMatched));
    matchIdx = matchIdx(isMatched);
end

imds.Labels = categorical(labelTable.diagnosis(matchIdx), 0:4);
fprintf('Total: %d images\n', numel(imds.Files));
disp(countcats(imds.Labels));

%% 3. STRATIFIED SPLIT
rng(42);
[imdsTrain, imdsRem] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal, imdsTest]  = splitEachLabel(imdsRem, 0.50, 'randomized');

fprintf('Train: %d | Val: %d | Test: %d\n', ...
    numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));

% Persist Val + Test splits (files + labels) for Phase 3B calibration and
% for anyone re-computing final test metrics later without retraining.
valFiles   = imdsVal.Files;   valLabels  = imdsVal.Labels;
testFiles  = imdsTest.Files;  testLabels = imdsTest.Labels;
save(fullfile(scriptDir, 'Phase2_ValidationSplit.mat'), ...
    'valFiles', 'valLabels', 'testFiles', 'testLabels');
fprintf('Val/Test split saved -> Phase2_ValidationSplit.mat\n');

%% 4. MANUAL OVERSAMPLING
trainLabels = imdsTrain.Labels;
classNames  = categories(trainLabels);
classCounts = countcats(trainLabels);
targetCount = max(classCounts);

fprintf('Balancing to %d per class...\n', targetCount);

allFiles  = imdsTrain.Files;
allLabels = trainLabels;

for i = 1:numel(classNames)
    idx = find(trainLabels == classNames{i});
    nCurrent = numel(idx);
    if nCurrent < targetCount && nCurrent > 0
        nExtra = targetCount - nCurrent;
        pickIdx = idx(randi(nCurrent, nExtra, 1));
        allFiles  = [allFiles;  imdsTrain.Files(pickIdx)];   %#ok<AGROW>
        allLabels = [allLabels; trainLabels(pickIdx)];        %#ok<AGROW>
    end
end

shuf = randperm(numel(allLabels));
imdsTrain = imageDatastore(allFiles(shuf), 'Labels', allLabels(shuf));
fprintf('Balanced: %d images\n', numel(imdsTrain.Files));

%% 5. IMAGENET NORMALIZATION (applied ONCE, in cpuRead - see input layer fix in step 6)
meanRGB = single([0.485 0.456 0.406]);
stdRGB  = single([0.229 0.224 0.225]);

imdsTrain.ReadFcn = @(loc) cpuRead(loc, CFG.inputSize(1:2), meanRGB, stdRGB, true);
imdsVal.ReadFcn   = @(loc) cpuRead(loc, CFG.inputSize(1:2), meanRGB, stdRGB, false);
imdsTest.ReadFcn  = @(loc) cpuRead(loc, CFG.inputSize(1:2), meanRGB, stdRGB, false);

%% 6. NETWORK SETUP + LAYER FREEZING
switch CFG.network
    case 'resnet50'
        net = resnet50;
        finalLayerName = 'fc1000';
        finalClassLayerName = 'ClassificationLayer_fc1000';
    case 'resnet18'
        net = resnet18;
        finalLayerName = 'fc1000';
        finalClassLayerName = 'ClassificationLayer_fc1000';
    otherwise
        error('CFG.network must be resnet50 or resnet18');
end

lgraph = layerGraph(net);
numClasses = 5;

% ---- FIX (critical): disable the network's own built-in normalization.
% cpuRead() already normalizes with ImageNet mean/std above; leaving the
% default imageInputLayer normalization ('zerocenter', expects raw 0-255
% pixels) active on top of that double-normalizes every image and badly
% distorts what the network actually sees during training.
inputLayerName = lgraph.Layers(1).Name;
newInput = imageInputLayer(CFG.inputSize, 'Name', inputLayerName, ...
    'Normalization', 'none');
lgraph = replaceLayer(lgraph, inputLayerName, newInput);

newFc = fullyConnectedLayer(numClasses, 'Name', 'fc_dr', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClass = classificationLayer('Name', 'classoutput', 'Classes', categories(imds.Labels));

lgraph = replaceLayer(lgraph, finalLayerName, newFc);
lgraph = replaceLayer(lgraph, finalClassLayerName, newClass);

% Freeze the first CFG.freezeUpToPercent of layers
allLayers = lgraph.Layers;
numLayers = numel(allLayers);
freezeCount = round(CFG.freezeUpToPercent * numLayers);

for i = 1:freezeCount
    L = allLayers(i);
    if isprop(L, 'WeightLearnRateFactor')
        L.WeightLearnRateFactor = 0;
    end
    if isprop(L, 'BiasLearnRateFactor')
        L.BiasLearnRateFactor = 0;
    end
    lgraph = replaceLayer(lgraph, L.Name, L);
end

fprintf('Frozen %d/%d layers (fine-tuning only the last %.0f%%)\n', ...
    freezeCount, numLayers, 100*(1-CFG.freezeUpToPercent));

%% 7. TRAINING OPTIONS
checkpointDir = fullfile(scriptDir, 'checkpoints');
if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end

trainOpts = { ...
    'MiniBatchSize', CFG.batchSize, ...
    'MaxEpochs', CFG.maxEpochs, ...
    'InitialLearnRate', CFG.initialLR, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 7, ...
    'LearnRateDropFactor', 0.25, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsVal, ...
    'ValidationFrequency', CFG.valFreq, ...
    'ValidationPatience', CFG.valPatience, ...
    'GradientThreshold', 1, ...
    'L2Regularization', 1e-4, ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu', ...
    'CheckpointPath', checkpointDir, ...
    'CheckpointFrequency', 10, ...
    'OutputNetwork', 'best-validation-loss'};

options = trainingOptions('adam', trainOpts{:});

fprintf('\n=== TRAINING START ===\n');
tic;
[trainedNet, trainInfo] = trainNetwork(imdsTrain, lgraph, options);
fprintf('Training done in %.1f minutes\n', toc/60);

%% 8. TTA INFERENCE
if CFG.useTTA
    fprintf('\n=== TTA Inference (%d views) ===\n', CFG.numTTA);
    predLabels = cpuTTA(trainedNet, imdsTest, CFG, meanRGB, stdRGB);
else
    predLabels = classify(trainedNet, imdsTest);
end

acc = mean(predLabels == imdsTest.Labels);
fprintf('\nTest Accuracy: %.2f%%\n', acc*100);

%% 9. CONFUSION MATRIX + PER-CLASS METRICS
figure('Name', 'DR Confusion', 'Position', [100 100 900 700]);
confusionchart(imdsTest.Labels, predLabels, ...
    'Title', sprintf('Acc: %.2f%% | %s | CPU | Batch: %d', acc*100, CFG.network, CFG.batchSize), ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

fprintf('\nPer-Class Sensitivity:\n');
cats = categories(imdsTest.Labels);
for i = 1:numel(cats)
    idx = imdsTest.Labels == cats{i};
    if any(idx)
        fprintf('   Grade %s: %.1f%%\n', cats{i}, mean(predLabels(idx)==cats{i})*100);
    end
end

%% 10. REFERABLE DR METRIC (Level 2+, required by problem statement)
% Grades 0,1 = non-referable; Grades 2,3,4 = referable
testGradeNum = double(imdsTest.Labels) - 1;  % categorical index -> actual grade 0-4
predGradeNum = double(predLabels) - 1;

isReferableTrue = testGradeNum >= 2;
isReferablePred = predGradeNum >= 2;

TP = sum(isReferableTrue & isReferablePred);
FN = sum(isReferableTrue & ~isReferablePred);
TN = sum(~isReferableTrue & ~isReferablePred);
FP = sum(~isReferableTrue & isReferablePred);

referableSens = TP / max(TP+FN, 1);
referableSpec = TN / max(TN+FP, 1);

fprintf('\nReferable DR (Level 2+) Screening Metric:\n');
fprintf('   Sensitivity: %.2f%%  (target >90%%)\n', referableSens*100);
fprintf('   Specificity: %.2f%%  (target >85%%)\n', referableSpec*100);

%% 11. SAVE
% Saved as RetinaAegis_Classifier.mat (not RetinaAegis_CPU.mat) so
% Phase 3 (Grad-CAM) and Phase 3B (calibration/uncertainty) can load it
% without any path changes. The FC layer is named 'fc_dr' here (not
% 'new_fc') - Phase 3B must read logits from 'fc_dr' to match.
save(fullfile(scriptDir, 'RetinaAegis_Classifier.mat'), ...
    'trainedNet', 'trainInfo', 'acc', 'referableSens', 'referableSpec', 'CFG', '-v7.3');
fprintf('\nSaved -> RetinaAegis_Classifier.mat\n');


%% ================================================================
% HELPER FUNCTIONS
%% ================================================================

function img = cpuRead(filePath, targetSize, meanRGB, stdRGB, doAugment)

    img = imread(filePath);

    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    elseif size(img,3) == 4
        img = img(:,:,1:3);
    end

    img = imresize(img, targetSize, 'bilinear');

    if doAugment
        if rand > 0.5
            img = img(:, end:-1:1, :);
        end
        if rand > 0.5
            img = img(end:-1:1, :, :);
        end
        theta = randi([-30 30]);
        if theta ~= 0
            img = imrotate(img, theta, 'bilinear', 'crop');
        end
    end

    img = im2single(img);
    img = (img - reshape(meanRGB,1,1,3)) ./ reshape(stdRGB,1,1,3);
end

function predLabels = cpuTTA(net, imds, CFG, meanRGB, stdRGB)

    nFiles = numel(imds.Files);
    nClasses = numel(categories(imds.Labels));
    scoreAcc = zeros(nFiles, nClasses, 'single');

    numChunks = ceil(nFiles / CFG.TTAchunk);

    for c = 1:numChunks
        tic;
        iStart = (c-1)*CFG.TTAchunk + 1;
        iEnd = min(c*CFG.TTAchunk, nFiles);
        nChunk = iEnd - iStart + 1;

        batchSize = nChunk * CFG.numTTA;
        batchImgs = zeros(224, 224, 3, batchSize, 'single');

        idx = 1;
        for i = iStart:iEnd
            img = imread(imds.Files{i});
            if size(img,3) == 1, img = repmat(img,[1 1 3]); end
            img = imresize(img, [224 224], 'bilinear');

            for j = 1:CFG.numTTA
                if j == 2
                    aug = img(:, end:-1:1, :);
                elseif j == 3
                    aug = img(end:-1:1, :, :);
                elseif j == 4
                    aug = imrotate(img, 10, 'bilinear', 'crop');
                elseif j == 5
                    aug = imrotate(img, -10, 'bilinear', 'crop');
                else
                    aug = img;
                end

                aug = im2single(aug);
                aug = (aug - reshape(meanRGB,1,1,3)) ./ reshape(stdRGB,1,1,3);
                batchImgs(:,:,:,idx) = aug;
                idx = idx + 1;
            end
        end

        scores = predict(net, batchImgs);

        for i = 1:nChunk
            s = (i-1)*CFG.numTTA + 1;
            e = i*CFG.numTTA;
            scoreAcc(iStart+i-1, :) = mean(scores(s:e,:), 1);
        end

        fprintf('  Chunk %d/%d (%.1fs)\n', c, numChunks, toc);
    end

    [~, predIdx] = max(scoreAcc, [], 2);
    predLabels = categorical(predIdx - 1, 0:4);
end