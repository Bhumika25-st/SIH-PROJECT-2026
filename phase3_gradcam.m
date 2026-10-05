%% Phase 3: Explainable AI (XAI) - Grad-CAM Heatmap Generation
% System: RETINA-AEGIS Module 3
% Function: Loads the trained CPU-optimized ResNet-50 classifier from
%           Phase 2, runs inference, and generates Grad-CAM attention
%           maps for the predicted DR grade (flowchart box 9).
%
%
% IMPORTANT - matches the CORRECTED Phase 2 (v7):
%   - trainedNet's input layer normalization is 'none' - it expects
%     ALREADY-NORMALIZED input (resize -> im2single -> ImageNet mean/std),
%     exactly like cpuRead() in Phase 2. Feeding gradCAM() a raw 0-255
%     image here would silently produce garbage attention maps, since
%     the network was never trained to see raw pixels.
%   - Final FC layer is named 'fc_dr' (not 'new_fc').
%   - Runs on imdsTest from Phase2_ValidationSplit.mat, since that's the
%     held-out set Phase 2 reported final accuracy/referable metrics on -
%     using the same set here keeps the Grad-CAM report and the reported
%     numbers about the same population of images.

clear; clc; close all;

%% 1. Configuration & Path Setup
scriptDir  = fileparts(mfilename('fullpath'));
modelFile  = fullfile(scriptDir, 'RetinaAegis_Classifier.mat');
splitFile  = fullfile(scriptDir, 'Phase2_ValidationSplit.mat');
outputDir  = fullfile(scriptDir, 'xai_reports', 'gradcam_results');
overlayDir = fullfile(outputDir, 'overlays');
rawMapDir  = fullfile(outputDir, 'raw_heatmaps');

if ~exist(outputDir, 'dir'),  mkdir(outputDir);  end
if ~exist(overlayDir, 'dir'), mkdir(overlayDir); end
if ~exist(rawMapDir, 'dir'),  mkdir(rawMapDir);  end

if ~exist(modelFile, 'file')
    error('Trained model not found: %s\nRun Phase 2 first.', modelFile);
end
if ~exist(splitFile, 'file')
    error('Split file not found: %s\nRun Phase 2 first.', splitFile);
end

FC_LAYER_NAME = 'fc_dr';   % must match Phase 2's newFc layer name
inputSize     = [224 224]; % matches CFG.inputSize(1:2) in Phase 2

% ImageNet normalization constants - MUST match cpuRead() in Phase 2 exactly
meanRGB = single([0.485 0.456 0.406]);
stdRGB  = single([0.229 0.224 0.225]);

gradeNames = {'0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR'};
OVERLAY_ALPHA = 0.5;

%% 2. Load Trained Network + Test Split
fprintf('Loading trained classifier from %s ...\n', modelFile);
M = load(modelFile, 'trainedNet');
trainedNet = M.trainedNet;

fprintf('Loading test split from %s ...\n', splitFile);
S = load(splitFile, 'testFiles', 'testLabels');
testFiles  = S.testFiles;
testLabels = S.testLabels;
numImages  = numel(testFiles);
fprintf('Generating Grad-CAM for %d test-set images.\n', numImages);

% Locate the last conv/ReLU layer dynamically - robust to resnet50 vs
% resnet18 (CFG.network in Phase 2 can be either).
gradCAMLayerName = findLastConvLayer(trainedNet);
fprintf('Using layer "%s" for Grad-CAM activation mapping.\n', gradCAMLayerName);

%% 3. Results Container (persisted for Phase 5 fusion)
gradcamResults = struct('Filename', {}, 'PredictedIdx', {}, 'PredictedGrade', {}, ...
    'TrueIdx', {}, 'Scores', {}, 'RawScoreMapFile', {}, 'OverlayFile', {});

reportTable = table('Size', [0 5], ...
    'VariableTypes', {'string', 'string', 'string', 'double', 'logical'}, ...
    'VariableNames', {'Filename', 'TrueGrade', 'PredictedGrade', 'RawConfidence', 'Correct'});

%% 4. Inference + Grad-CAM Loop
for k = 1:numImages
    filePath = testFiles{k};
    [~, baseName, ext] = fileparts(filePath);
    trueIdx = double(testLabels(k)) - 1;

    % Normalize EXACTLY like cpuRead() (no augmentation at inference time)
    imgNorm = readAndNormalize(filePath, inputSize, meanRGB, stdRGB);

    %% Step A: Classification (raw, uncalibrated)
    scores = predict(trainedNet, imgNorm);
    [rawConf, predictedIdxOneBased] = max(scores);
    predictedIdx = predictedIdxOneBased - 1;
    gradeStr = gradeNames{predictedIdx + 1};
    trueGradeStr = gradeNames{trueIdx + 1};

    % gradCAM() needs a categorical/label argument for the class to
    % explain - build it against the network's known class set.
    predictedLabel = categorical(predictedIdx, 0:4);

    %% Step B: Grad-CAM Map for the Predicted Class
    % NOTE: imgNorm is already normalized exactly as the network expects
    % (matches training-time cpuRead preprocessing minus augmentation).
    scoreMap = gradCAM(trainedNet, imgNorm, predictedLabel, ...
        'FeatureLayer', gradCAMLayerName);
    scoreMapResized = imresize(rescale(scoreMap), inputSize);

    %% Step C: Save Raw Heatmap Data (for fusion with lesion masks in Phase 5)
    rawMapFile = fullfile(rawMapDir, [baseName '_gradcam.mat']);
    save(rawMapFile, 'scoreMapResized');

    %% Step D: Save Visual Overlay
    % For DISPLAY only, reconstruct a viewable RGB image (undo the
    % ImageNet normalization) - gradCAM/the network never see this
    % version, it's purely for the human-readable overlay PNG.
    displayImg = denormalizeForDisplay(imgNorm, meanRGB, stdRGB);

    % Compute a field-of-view mask (retina vs black background) and
    % clip the heatmap to it. Grad-CAM's coarse spatial resolution
    % (~7x7 for ResNet-50's last conv layer, upsampled 32x to 224x224)
    % means a single activated cell can span both retina and background
    % simultaneously - without this mask, that shows up as heatmap
    % color bleeding visibly outside the retinal circle.
    fovMask = computeFOVMask(rgb2gray(displayImg));

    overlayImg = createGradCAMOverlay(displayImg, scoreMapResized, fovMask, OVERLAY_ALPHA);
    overlayFile = fullfile(overlayDir, [baseName '_gradcam.png']);
    imwrite(overlayImg, overlayFile);

    %% Step E: Log
    gradcamResults(end+1) = struct( ...
        'Filename', string(baseName) + string(ext), ...
        'PredictedIdx', predictedIdx, ...
        'PredictedGrade', string(gradeStr), ...
        'TrueIdx', trueIdx, ...
        'Scores', scores, ...
        'RawScoreMapFile', string(rawMapFile), ...
        'OverlayFile', string(overlayFile)); %#ok<SAGROW>

    reportTable = [reportTable; {string(baseName) + string(ext), string(trueGradeStr), ...
        string(gradeStr), rawConf*100, predictedIdx == trueIdx}]; %#ok<AGROW>

    if mod(k, 25) == 0 || k == numImages
        fprintf('Processed %d / %d images...\n', k, numImages);
    end
end

%% 5. Persist Results for Downstream Phases (5)
save(fullfile(outputDir, 'Phase3_GradCAM_Results.mat'), 'gradcamResults');
writetable(reportTable, fullfile(outputDir, 'Phase3_GradCAM_Report.csv'));

overallAcc = mean(reportTable.Correct);
fprintf('\nExecution Completed:\n');
fprintf('  Images Processed: %d\n', numImages);
fprintf('  Test-set Accuracy (sanity check vs Phase 2): %.2f%%\n', overallAcc*100);
fprintf('  Grad-CAM overlays saved to: %s\n', overlayDir);
fprintf('  Raw heatmaps saved to: %s\n', rawMapDir);
fprintf('  Results struct saved to: %s\n', fullfile(outputDir, 'Phase3_GradCAM_Results.mat'));


%% ================================================================
% HELPER FUNCTIONS
%% ================================================================

function layerName = findLastConvLayer(net)
    layers = net.Layers;
    layerName = '';
    for i = 1:numel(layers)
        if isa(layers(i), 'nnet.cnn.layer.ReLULayer') || ...
           isa(layers(i), 'nnet.cnn.layer.Convolution2DLayer')
            layerName = layers(i).Name;
        end
    end
    if isempty(layerName)
        error('Could not automatically locate a convolutional/ReLU layer for Grad-CAM.');
    end
end

function img = readAndNormalize(filePath, targetSize, meanRGB, stdRGB)
    % Mirrors cpuRead() from Phase 2 EXACTLY (minus augmentation, which
    % should never be applied at inference time).
    img = imread(filePath);
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    elseif size(img,3) == 4
        img = img(:,:,1:3);
    end
    img = imresize(img, targetSize, 'bilinear');
    img = im2single(img);
    img = (img - reshape(meanRGB,1,1,3)) ./ reshape(stdRGB,1,1,3);
end

function displayImg = denormalizeForDisplay(imgNorm, meanRGB, stdRGB)
    % Reverses cpuRead's normalization so humans can actually see the
    % fundus image under the heatmap overlay. Never fed back into the
    % network - display purposes only.
    imgDenorm = imgNorm .* reshape(stdRGB,1,1,3) + reshape(meanRGB,1,1,3);
    imgDenorm = max(min(imgDenorm, 1), 0); % clip to valid [0,1] range
    displayImg = im2uint8(imgDenorm);
end

function overlayImg = createGradCAMOverlay(originalImg, scoreMapResized, fovMask, alpha)
    cmap = jet(256);
    heatmapIdx = uint8(scoreMapResized * 255) + 1;
    heatmapRGB = ind2rgb(heatmapIdx, cmap);
    heatmapRGB = im2uint8(heatmapRGB);

    blended = uint8((1 - alpha) * double(originalImg) + alpha * double(heatmapRGB));

    % Outside the retinal field of view, show the original image
    % untouched - never the heatmap color. This is what actually stops
    % the visible "bleeding" outside the expected retinal area.
    overlayImg = originalImg;
    for c = 1:3
        channel = overlayImg(:,:,c);
        blendedChannel = blended(:,:,c);
        channel(fovMask) = blendedChannel(fovMask);
        overlayImg(:,:,c) = channel;
    end
end

function fovMask = computeFOVMask(grayImg)
    % Field-of-view mask: separates the circular/elliptical retina from
    % the black background surrounding it in a standard fundus photo.
    % Same approach as Phase 4's FOV detection, kept independent here
    % since Phase 3 and Phase 4 are meant to run standalone.
    fovMask = grayImg > 15;
    fovMask = imfill(fovMask, 'holes');
    cc = bwconncomp(fovMask);
    if cc.NumObjects > 0
        areas = cellfun(@numel, cc.PixelIdxList);
        [~, biggest] = max(areas);
        fovMask = false(size(fovMask));
        fovMask(cc.PixelIdxList{biggest}) = true;
    end
    fovMask = imerode(fovMask, strel('disk', 5)); % pull in from the rim
end