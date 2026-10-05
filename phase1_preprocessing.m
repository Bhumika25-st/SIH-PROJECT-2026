%% RETINA-AEGIS Module 1 (v5 - ML-based, properly validated)
%
% MAJOR REDESIGN from v4.2, addressing reviewer feedback:
%   1. Hand-written weighted score REMOVED -> replaced by a trained
%      classifier (logistic regression via fitclinear, or an ensemble
%      via fitcensemble -- both tried, best one kept).
%   2. ROI fallback improved -- instead of "give up and use whole
%      image", tries multiple Otsu-style thresholds and a circular
%      Hough-transform-based disk fit before giving up.
%   3. Vessel density kept only as ONE input feature among many --
%      no hardcoded weight. If it's unreliable, the classifier will
%      learn to downweight it on its own from your labeled data.
%   4. Added block-wise local illumination/contrast features (a grid
%      of patches) instead of relying only on global mean intensity --
%      catches localized glare, shadow, or artifact regions.
%   5. Proper train/validation/test split (60/20/20) with AUC,
%      sensitivity, specificity, F1, balanced accuracy, and a
%      confusion matrix reported on a held-out TEST set the model
%      never saw during training or threshold selection.
%
% WORKFLOW (run in this order, changing config.MODE each time):
%   Step 1: config.MODE = 'extract'  -> computes features for every
%           image, saves them to Features_v5.csv. Run this once.
%   Step 2: Manually label a subset (~150-300 images recommended) as
%           gradeable(1)/ungradeable(0) in a CSV with columns
%           Filename,Label. Point config.LABELED_CSV at it.
%   Step 3: config.MODE = 'train'    -> loads Features_v5.csv, joins
%           with your labels, does the train/val/test split, trains
%           the classifier, prints/plots all metrics, saves the model
%           to RetinaQualityModel.mat.
%   Step 4: config.MODE = 'infer'    -> loads the saved model, scores
%           every image (including unlabeled ones), applies the
%           validation-derived PASS/REVIEW thresholds, and writes
%           actual output images (CLAHE-enhanced for PASS) to disk.

clear; clc; close all;

%% ================================================================
% 1. CONFIGURATION
%% ================================================================

config.inputDir   = fullfile('.', 'aptos2019-blindness-detection', 'train_images');
config.outputDir  = fullfile('.', 'aptos2019-blindness-detection', 'processed_v5');

config.passDir    = fullfile(config.outputDir, 'gradeable');
config.reviewDir  = fullfile(config.outputDir, 'review');
config.rejectDir  = fullfile(config.outputDir, 'rejected');

dirs = {config.outputDir, config.passDir, config.reviewDir, config.rejectDir};
for i = 1:length(dirs)
    if ~exist(dirs{i}, 'dir'), mkdir(dirs{i}); end
end

config.USE_MAX_PERFORMANCE = false;

config.QUALITY_SIZE = [256 256];
config.VESSEL_SIZE  = [512 512];
config.TARGET_SIZE  = [512 512];

% ---- MODE ----
% 'extract'    -> compute features only (no labels needed)
% 'train'      -> train + evaluate classifier (needs LABELED_CSV)
% 'infer'      -> apply TRAINED model to full dataset, save outputs
% 'quick_sort' -> NO training/labels needed. Reads Features_v5.csv
%                 (from a previous 'extract' run) and sorts images
%                 into gradeable/review/rejected RIGHT NOW using a
%                 simple heuristic score built from the raw features.
%                 Use this if you want folders filled immediately and
%                 will do the proper ML calibration ('train'/'infer')
%                 later for your final validated submission.
config.MODE = 'quick_sort';

config.FEATURES_CSV = fullfile(config.outputDir, 'Features_v5.csv');
config.LABELED_CSV  = '';   % e.g. 'my_manual_labels.csv' — Filename,Label
config.MODEL_PATH   = fullfile(config.outputDir, 'RetinaQualityModel.mat');

% Hard-fail-only clipping thresholds still apply pre-classifier
% (a totally blown-out image is a real hard failure, not a judgment
% call for the classifier to make)
config.MAX_DARK_RATIO   = 0.35;
config.MAX_BRIGHT_RATIO = 0.05;

% Grid size for block-wise local illumination features (point 4)
config.ILLUM_GRID = [4 4];   % 4x4 = 16 patches


%% ================================================================
% 2. FIND IMAGES
%% ================================================================

supportedExts = {'*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'};
imgFiles = [];
for ext = supportedExts
    files = dir(fullfile(config.inputDir, ext{1}));
    imgFiles = [imgFiles; files]; %#ok<AGROW>
end
numFiles = length(imgFiles);
fprintf('Found %d images.\n', numFiles);

if config.USE_MAX_PERFORMANCE && isempty(gcp('nocreate'))
    fprintf('Starting parallel pool...\n');
    parpool('local');
end


%% ================================================================
% 3. MODE: EXTRACT
% Computes raw features for every image -- NO scoring/weighting here.
% This is the key structural change: feature extraction and decision-
% making are now fully separated.
%% ================================================================

if strcmp(config.MODE, 'extract')

    featResults(numFiles) = emptyFeatureStruct();

    q = parallel.pool.DataQueue;
    afterEach(q, @updateProgress);

    if config.USE_MAX_PERFORMANCE
        parfor k = 1:numFiles
            featResults(k) = extractFeatures(imgFiles(k), config);
            send(q, 1);
        end
    else
        for k = 1:numFiles
            featResults(k) = extractFeatures(imgFiles(k), config);
            if mod(k,10)==0, fprintf('  %d/%d\n', k, numFiles); end
        end
    end

    featTable = struct2table(featResults);
    writetable(featTable, config.FEATURES_CSV);

    fprintf('\n✅ Features extracted for %d images -> %s\n', numFiles, config.FEATURES_CSV);
    fprintf('Next: label a subset (Filename,Label CSV) and set MODE=''train''.\n');
    return;
end


%% ================================================================
% 4. MODE: TRAIN
% Loads features + your manual labels, does a proper train/val/test
% split, trains a classifier, and reports real metrics.
%% ================================================================

if strcmp(config.MODE, 'train')

    if isempty(config.LABELED_CSV) || ~exist(config.LABELED_CSV, 'file')
        error('config.LABELED_CSV must point to a Filename,Label CSV to train.');
    end
    if ~exist(config.FEATURES_CSV, 'file')
        error('Run MODE=''extract'' first to generate %s', config.FEATURES_CSV);
    end

    featTable = readtable(config.FEATURES_CSV);
    labelTable = readtable(config.LABELED_CSV);

    [tf, idx] = ismember(string(featTable.Filename), string(labelTable.Filename));
    if sum(tf) < 30
        error('Only %d labeled images matched. Label at least ~100-150 for a meaningful split.', sum(tf));
    end

    dataTable = featTable(tf, :);
    dataTable.Label = labelTable.Label(idx(tf));

    fprintf('Matched %d labeled images (Gradeable=%d, Ungradeable=%d)\n', ...
        height(dataTable), sum(dataTable.Label==1), sum(dataTable.Label==0));

    featureNames = {'Tenengrad','Laplacian','MeanIntensity','Contrast', ...
        'DarkRatio','BrightRatio','IllumVariation','VesselDensity', ...
        'LocalIllumStd','LocalContrastStd','MaxPatchGlareFrac'};

    X = table2array(dataTable(:, featureNames));
    y = dataTable.Label;

    % Standardize features (important for logistic regression / SVM)
    muX = mean(X, 1);
    sigX = std(X, 0, 1);
    sigX(sigX == 0) = 1;
    Xz = (X - muX) ./ sigX;

    %% ---- 60/20/20 TRAIN/VAL/TEST SPLIT ----
    rng(42); % reproducible split
    cvOuter = cvpartition(y, 'HoldOut', 0.40);   % 60% train, 40% temp
    trainIdx = training(cvOuter);
    tempIdx  = test(cvOuter);

    tempY = y(tempIdx);
    cvInner = cvpartition(tempY, 'HoldOut', 0.50); % split temp 50/50 -> val/test
    tempIdxFull = find(tempIdx);
    valIdx  = false(size(y)); valIdx(tempIdxFull(training(cvInner))) = true;
    testIdx = false(size(y)); testIdx(tempIdxFull(test(cvInner))) = true;

    Xtrain = Xz(trainIdx,:); ytrain = y(trainIdx);
    Xval   = Xz(valIdx,:);   yval   = y(valIdx);
    Xtest  = Xz(testIdx,:);  ytest  = y(testIdx);

    fprintf('Split -> Train: %d | Val: %d | Test: %d\n', ...
        sum(trainIdx), sum(valIdx), sum(testIdx));

    %% ---- TRAIN CLASSIFIER ----
    % Try both a logistic regression (interpretable, simple) and a
    % random-forest-style ensemble (usually stronger with mixed-scale
    % features), pick whichever does better on the VALIDATION set.

    mdlLogistic = fitclinear(Xtrain, ytrain, 'Learner', 'logistic');
    mdlEnsemble = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', 'NumLearningCycles', 100);

    [~, scoreLogVal] = predict(mdlLogistic, Xval);
    [~, scoreEnsVal]  = predict(mdlEnsemble, Xval);

    aucLogVal = aucFromScores(yval, scoreLogVal(:,2));
    aucEnsVal = aucFromScores(yval, scoreEnsVal(:,2));

    fprintf('\nValidation AUC -- Logistic: %.3f | Ensemble: %.3f\n', aucLogVal, aucEnsVal);

    if aucEnsVal >= aucLogVal
        bestModel = mdlEnsemble;
        bestName = 'Ensemble (Bagged Trees)';
    else
        bestModel = mdlLogistic;
        bestName = 'Logistic Regression';
    end
    fprintf('Selected model: %s\n', bestName);

    %% ---- CHOOSE THRESHOLDS ON VALIDATION SET (never on test) ----
    [~, scoreVal] = predict(bestModel, Xval);
    probVal = scoreVal(:,2);

    [rocX, rocY, rocT, aucVal] = perfcurve(yval, probVal, 1);
    J = rocY - rocX;
    [~, bestJ] = max(J);
    passThresh = rocT(bestJ);

    sensTargetIdx = find(rocY >= 0.90, 1, 'first');
    if isempty(sensTargetIdx)
        reviewThresh = passThresh * 0.6;
    else
        reviewThresh = rocT(sensTargetIdx);
    end
    reviewThresh = min(reviewThresh, passThresh - 0.01); % ensure ordering

    fprintf('\nThresholds chosen on VALIDATION set:\n');
    fprintf('   PASS_PROB_THRESH   = %.3f\n', passThresh);
    fprintf('   REVIEW_PROB_THRESH = %.3f\n', reviewThresh);

    %% ---- FINAL EVALUATION ON HELD-OUT TEST SET ----
    [predTest, scoreTest] = predict(bestModel, Xtest);
    probTest = scoreTest(:,2);

    [~, ~, ~, aucTest] = perfcurve(ytest, probTest, 1);

    predBinary = probTest >= passThresh;

    confMat = confusionmat(ytest, predBinary, 'Order', [1 0]);
    TP = confMat(1,1); FN = confMat(1,2);
    FP = confMat(2,1); TN = confMat(2,2);

    sensitivity = TP / max(TP+FN, 1);       % recall / true positive rate
    specificity = TN / max(TN+FP, 1);
    precision   = TP / max(TP+FP, 1);
    f1          = 2 * (precision*sensitivity) / max(precision+sensitivity, eps);
    balancedAcc = (sensitivity + specificity) / 2;

    fprintf('\n============================================\n');
    fprintf(' TEST SET METRICS (held-out, never used for training/thresholds)\n');
    fprintf('============================================\n');
    fprintf('AUC                 : %.3f\n', aucTest);
    fprintf('Sensitivity (Recall): %.3f\n', sensitivity);
    fprintf('Specificity         : %.3f\n', specificity);
    fprintf('Precision           : %.3f\n', precision);
    fprintf('F1 Score            : %.3f\n', f1);
    fprintf('Balanced Accuracy   : %.3f\n', balancedAcc);
    fprintf('\nConfusion Matrix (rows=actual, cols=predicted) [Gradeable, Ungradeable]:\n');
    disp(array2table(confMat, ...
        'VariableNames', {'Pred_Gradeable','Pred_Ungradeable'}, ...
        'RowNames', {'Actual_Gradeable','Actual_Ungradeable'}));

    figure('Name', 'ROC Curve (Test Set)');
    [rocXt, rocYt] = perfcurve(ytest, probTest, 1);
    plot(rocXt, rocYt, 'b-', 'LineWidth', 2); hold on;
    plot([0 1],[0 1],'k--');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('Test Set ROC (AUC = %.3f) — %s', aucTest, bestName));
    grid on;

    %% ---- SAVE MODEL + THRESHOLDS + NORMALIZATION ----
    save(config.MODEL_PATH, 'bestModel', 'bestName', 'muX', 'sigX', ...
        'featureNames', 'passThresh', 'reviewThresh', ...
        'aucTest', 'sensitivity', 'specificity', 'f1', 'balancedAcc');

    fprintf('\n✅ Model + calibrated thresholds saved -> %s\n', config.MODEL_PATH);
    fprintf('Next: set MODE=''infer'' to score and save the full dataset.\n');
    return;
end


%% ================================================================
% 5. MODE: INFER
% Applies the trained model to the full dataset and writes outputs.
%% ================================================================

if strcmp(config.MODE, 'infer')

    if ~exist(config.MODEL_PATH, 'file')
        error('No trained model found at %s. Run MODE=''train'' first.', config.MODEL_PATH);
    end
    if ~exist(config.FEATURES_CSV, 'file')
        error('Run MODE=''extract'' first to generate %s', config.FEATURES_CSV);
    end

    M = load(config.MODEL_PATH);
    featTable = readtable(config.FEATURES_CSV);

    X = table2array(featTable(:, M.featureNames));
    Xz = (X - M.muX) ./ M.sigX;

    [~, scores] = predict(M.bestModel, Xz);
    prob = scores(:,2);

    status = strings(height(featTable), 1);
    for i = 1:height(featTable)
        row = featTable(i,:);
        if row.DarkRatio > config.MAX_DARK_RATIO || row.BrightRatio > config.MAX_BRIGHT_RATIO
            status(i) = "REJECT_RECAPTURE";
        elseif prob(i) >= M.passThresh
            status(i) = "PASS_GRADEABLE";
        elseif prob(i) >= M.reviewThresh
            status(i) = "REVIEW";
        else
            status(i) = "REJECT_RECAPTURE";
        end
    end

    featTable.ModelProbability = prob;
    featTable.Status = status;
    writetable(featTable, fullfile(config.outputDir, 'Final_Assessment_v5.csv'));

    fprintf('Model: %s | Test AUC was %.3f, Sensitivity %.3f, Specificity %.3f\n', ...
        M.bestName, M.aucTest, M.sensitivity, M.specificity);
    fprintf('Gradeable: %d | Review: %d | Reject: %d\n', ...
        sum(status=="PASS_GRADEABLE"), sum(status=="REVIEW"), sum(status=="REJECT_RECAPTURE"));

    % Now actually save enhanced/review/reject images
    q = parallel.pool.DataQueue;
    afterEach(q, @updateProgress);

    if config.USE_MAX_PERFORMANCE
        parfor k = 1:numFiles
            saveOneOutput(imgFiles(k), status(k), config);
            send(q, 1);
        end
    else
        for k = 1:numFiles
            saveOneOutput(imgFiles(k), status(k), config);
        end
    end

    fprintf('\n✅ Inference complete. Images written to %s\n', config.outputDir);
    return;
end

%% ================================================================
% 6. MODE: QUICK_SORT
% No training/labels needed. Reads the CSV from a previous 'extract'
% run and applies a simple heuristic score (same idea as v4.2) to
% sort images into folders RIGHT NOW. This is a stop-gap so you have
% something to show immediately -- it is NOT the validated classifier
% from 'train'/'infer', and should be replaced by that for your final
% submission once you've labeled a subset.
%% ================================================================

if strcmp(config.MODE, 'quick_sort')

    if ~exist(config.FEATURES_CSV, 'file')
        error('Run MODE=''extract'' first to generate %s', config.FEATURES_CSV);
    end

    featTable = readtable(config.FEATURES_CSV);

    % ---- Heuristic normalization (same spirit as v4.2, unvalidated) ----
    sharpnessTenengrad = min(featTable.Tenengrad / 2500, 1);
    sharpnessLaplacian = min(featTable.Laplacian / 100, 1);
    sharpnessScore = 0.7*sharpnessTenengrad + 0.3*sharpnessLaplacian;

    illum = featTable.MeanIntensity;
    illuminationScore = zeros(height(featTable),1);
    illuminationScore(illum >= 30 & illum <= 210) = 1;
    lowMask = illum < 30;
    illuminationScore(lowMask) = max(0, illum(lowMask) / 30);
    highMask = illum > 210;
    illuminationScore(highMask) = max(0, (255 - illum(highMask)) / (255 - 210));

    contrastNormalized = min(featTable.Contrast / 60, 1);

    darkPenalty = min(featTable.DarkRatio / config.MAX_DARK_RATIO, 1);
    brightPenalty = min(featTable.BrightRatio / config.MAX_BRIGHT_RATIO, 1);
    clippingScore = max(0, 1 - 0.5*darkPenalty - 0.5*brightPenalty);

    uniformityScore = max(0, 1 - featTable.IllumVariation / 80);

    vesselScore = min(featTable.VesselDensity / 0.02, 1);

    % Local glare penalty (new v5 features) -- reduces score if a big
    % chunk of the image is flagged as flat/glare patches
    glarePenalty = max(0, 1 - featTable.MaxPatchGlareFrac);

    quickScore = ...
        0.35 * sharpnessScore + ...
        0.20 * illuminationScore + ...
        0.15 * contrastNormalized + ...
        0.10 * clippingScore + ...
        0.10 * uniformityScore + ...
        0.05 * vesselScore + ...
        0.05 * glarePenalty;

    PASS_SCORE = 0.65;
    REVIEW_SCORE = 0.45;

    status = strings(height(featTable), 1);
    isSeverelyClipped = featTable.DarkRatio > config.MAX_DARK_RATIO | ...
                         featTable.BrightRatio > config.MAX_BRIGHT_RATIO;

    status(isSeverelyClipped) = "REJECT_RECAPTURE";
    status(~isSeverelyClipped & quickScore >= PASS_SCORE) = "PASS_GRADEABLE";
    status(~isSeverelyClipped & quickScore >= REVIEW_SCORE & quickScore < PASS_SCORE) = "REVIEW";
    status(status == "") = "REJECT_RECAPTURE";

    featTable.QuickScore = quickScore;
    featTable.Status = status;
    writetable(featTable, fullfile(config.outputDir, 'QuickSort_Assessment.csv'));

    fprintf('Gradeable: %d | Review: %d | Reject: %d\n', ...
        sum(status=="PASS_GRADEABLE"), sum(status=="REVIEW"), sum(status=="REJECT_RECAPTURE"));
    fprintf('\n⚠️  This is an UNVALIDATED heuristic sort -- not the trained classifier.\n');
    fprintf('   For your final submission, label a subset and run MODE=''train''/''infer''.\n\n');

    % ---- Actually save the images now ----
    q = parallel.pool.DataQueue;
    afterEach(q, @updateProgress);

    if config.USE_MAX_PERFORMANCE
        parfor k = 1:numFiles
            saveOneOutput(imgFiles(k), status(k), config);
            send(q, 1);
        end
    else
        for k = 1:numFiles
            saveOneOutput(imgFiles(k), status(k), config);
        end
    end

    fprintf('✅ Images sorted into gradeable/review/rejected under %s\n', config.outputDir);
    return;
end


error('Unknown config.MODE: %s (use extract / train / infer / quick_sort)', config.MODE);


%% ================================================================
% HELPER: empty feature struct (for parfor preallocation)
%% ================================================================
function s = emptyFeatureStruct()
    s = struct('Filename', "", 'Tenengrad', 0, 'Laplacian', 0, ...
        'MeanIntensity', 0, 'Contrast', 0, 'DarkRatio', 0, ...
        'BrightRatio', 0, 'IllumVariation', 0, 'VesselDensity', 0, ...
        'LocalIllumStd', 0, 'LocalContrastStd', 0, 'MaxPatchGlareFrac', 0, ...
        'Status', "OK");
end


%% ================================================================
% HELPER: updateProgress callback
%% ================================================================
function updateProgress(~)
    persistent count
    if isempty(count), count = 0; end
    count = count + 1;
    if mod(count, 200) == 0
        fprintf('  -> %d processed...\n', count);
    end
end


%% ================================================================
% HELPER: aucFromScores
%% ================================================================
function a = aucFromScores(y, scores)
    [~,~,~,a] = perfcurve(y, scores, 1);
end


%% ================================================================
% IMPROVED ROI DETECTION (point 2)
% Instead of "give up -> use whole image" as the only fallback, tries:
%   1. Simple intensity threshold + morphology (as before)
%   2. If that fails, Otsu's method (adaptive, handles varying black
%      levels across cameras better than a fixed threshold of 10)
%   3. If that ALSO fails, Hough-circle fit for the retinal disk
%      (fundus images are always roughly circular/elliptical -- this
%      is a much better-justified fallback than "just use everything")
%   4. Only as a last resort, use the whole image (and this is now
%      logged in the Status field so you can audit how often it
%      happens -- if it's frequent, that itself is a data-quality flag)
%% ================================================================
function [roiMask, roiMethod] = detectROI(grayImg)

    % Attempt 1: fixed threshold + morphology
    roiMask = grayImg > 10;
    roiMask = bwareaopen(roiMask, 100);
    se = strel('disk', 5);
    roiMask = imclose(roiMask, se);
    roiMask = imfill(roiMask, 'holes');
    cc = bwconncomp(roiMask);
    if cc.NumObjects > 0
        areas = cellfun(@numel, cc.PixelIdxList);
        [~, largestIdx] = max(areas);
        roiMask = false(size(roiMask));
        roiMask(cc.PixelIdxList{largestIdx}) = true;
    end
    if nnz(roiMask) / numel(roiMask) >= 0.15
        roiMethod = "FixedThreshold";
        return;
    end

    % Attempt 2: Otsu's adaptive threshold
    try
        level = graythresh(grayImg);
        roiMask2 = imbinarize(grayImg, level);
        roiMask2 = bwareaopen(roiMask2, 100);
        roiMask2 = imclose(roiMask2, se);
        roiMask2 = imfill(roiMask2, 'holes');
        cc2 = bwconncomp(roiMask2);
        if cc2.NumObjects > 0
            areas2 = cellfun(@numel, cc2.PixelIdxList);
            [~, largestIdx2] = max(areas2);
            roiMask2 = false(size(roiMask2));
            roiMask2(cc2.PixelIdxList{largestIdx2}) = true;
        end
        if nnz(roiMask2) / numel(roiMask2) >= 0.15
            roiMask = roiMask2;
            roiMethod = "OtsuThreshold";
            return;
        end
    catch
        % fall through to next attempt
    end

    % Attempt 3: circular Hough transform for the retinal disk
    try
        minR = round(0.30 * min(size(grayImg)));
        maxR = round(0.55 * min(size(grayImg)));
        [centers, radii] = imfindcircles(grayImg, [minR maxR], ...
            'ObjectPolarity', 'bright', 'Sensitivity', 0.9);
        if ~isempty(centers)
            [~, biggest] = max(radii);
            [xx, yy] = meshgrid(1:size(grayImg,2), 1:size(grayImg,1));
            roiMask3 = ((xx - centers(biggest,1)).^2 + (yy - centers(biggest,2)).^2) ...
                <= radii(biggest)^2;
            if nnz(roiMask3) / numel(roiMask3) >= 0.15
                roiMask = roiMask3;
                roiMethod = "HoughCircle";
                return;
            end
        end
    catch
        % fall through to last resort
    end

    % Last resort: whole image (now explicitly logged, not silent)
    roiMask = true(size(grayImg));
    roiMethod = "FullImageFallback";
end


%% ================================================================
% FEATURE EXTRACTION (no scoring/weighting -- raw numbers only)
%% ================================================================
function r = extractFeatures(fileEntry, config)

    r = emptyFeatureStruct();
    r.Filename = string(fileEntry.name);

    try
        fullFilePath = fullfile(fileEntry.folder, fileEntry.name);
        [img, map] = imread(fullFilePath);

        if ~isempty(map)
            img = ind2rgb(img, map);
        end
        if ndims(img) == 2
            img = repmat(img, [1 1 3]);
        elseif size(img,3) == 4
            img = img(:,:,1:3);
        end
        if ~isa(img, 'uint8')
            img = im2uint8(img);
        end

        %% ---- QUALITY THUMBNAIL ----
        qualityImg = imresize(img, config.QUALITY_SIZE);
        grayImg = rgb2gray(qualityImg);
        grayDouble = double(grayImg);

        %% ---- IMPROVED ROI (point 2) ----
        [roiMask, roiMethod] = detectROI(grayImg);

        %% ---- SHARPNESS ----
        Gx = imfilter(grayDouble, [-1 0 1; -2 0 2; -1 0 1], 'replicate');
        Gy = imfilter(grayDouble, [-1 -2 -1; 0 0 0; 1 2 1], 'replicate');
        tenengradScore = mean(Gx(roiMask).^2 + Gy(roiMask).^2);

        lapFilter = [0 1 0; 1 -4 1; 0 1 0];
        lapResp = imfilter(grayDouble, lapFilter, 'replicate');
        laplacianScore = var(lapResp(roiMask));

        %% ---- GLOBAL ILLUMINATION / CONTRAST ----
        roiPixels = grayDouble(roiMask);
        meanIntensity = mean(roiPixels);
        contrastScore = std(roiPixels);
        darkRatio = mean(roiPixels < 15);
        brightRatio = mean(roiPixels > 245);

        localMean = imboxfilt(grayDouble, 31);
        illumVariation = std(localMean(roiMask));

        %% ---- BLOCK-WISE LOCAL FEATURES (point 4) ----
        [gh, gw] = size(grayDouble);
        gridR = config.ILLUM_GRID(1);
        gridC = config.ILLUM_GRID(2);
        rEdges = round(linspace(1, gh+1, gridR+1));
        cEdges = round(linspace(1, gw+1, gridC+1));

        patchMeans = [];
        patchStds  = [];
        glareCount = 0;
        validPatchCount = 0;

        for pr = 1:gridR
            for pc = 1:gridC
                rowRange = rEdges(pr):(rEdges(pr+1)-1);
                colRange = cEdges(pc):(cEdges(pc+1)-1);
                patch = grayDouble(rowRange, colRange);
                patchROI = roiMask(rowRange, colRange);
                if nnz(patchROI) < 20
                    continue; % patch mostly outside retina, skip
                end
                validPatchCount = validPatchCount + 1;
                pPixels = patch(patchROI);
                patchMeans(end+1) = mean(pPixels); %#ok<AGROW>
                patchStds(end+1) = std(pPixels); %#ok<AGROW>
                if mean(pPixels) > 230 && std(pPixels) < 10
                    glareCount = glareCount + 1;
                end
            end
        end

        if isempty(patchMeans)
            localIllumStd = 0;
            localContrastStd = 0;
            maxPatchGlareFrac = 0;
        else
            localIllumStd = std(patchMeans);
            localContrastStd = std(patchStds);
            maxPatchGlareFrac = glareCount / max(validPatchCount, 1);
        end

        %% ---- VESSEL DENSITY (point 3: kept as ONE feature, no fixed weight) ----
        vesselImg = imresize(img, config.VESSEL_SIZE);
        vesselGray = rgb2gray(vesselImg);
        seLineH = strel('line', 15, 0);
        seLineV = strel('line', 15, 90);
        tophatH = imtophat(vesselGray, seLineH);
        tophatV = imtophat(vesselGray, seLineV);
        vesselResponse = max(double(tophatH), double(tophatV));
        vesselMask = vesselResponse > (mean(vesselResponse(:)) + 0.5*std(vesselResponse(:)));
        vesselMask = bwareaopen(vesselMask, 5);
        vesselROI = imresize(uint8(roiMask), config.VESSEL_SIZE) > 0.5;
        vesselMask = vesselMask & vesselROI;
        vesselDensity = nnz(vesselMask) / max(nnz(vesselROI), 1);

        %% ---- POPULATE (raw features only -- no weighting/scoring here) ----
        r.Tenengrad = tenengradScore;
        r.Laplacian = laplacianScore;
        r.MeanIntensity = meanIntensity;
        r.Contrast = contrastScore;
        r.DarkRatio = darkRatio;
        r.BrightRatio = brightRatio;
        r.IllumVariation = illumVariation;
        r.VesselDensity = vesselDensity;
        r.LocalIllumStd = localIllumStd;
        r.LocalContrastStd = localContrastStd;
        r.MaxPatchGlareFrac = maxPatchGlareFrac;
        r.Status = roiMethod; % logs which ROI method was used (audit trail)

    catch ME
        r.Status = "ERROR: " + string(ME.message);
        warning('Failed: %s | %s', fileEntry.name, ME.message);
    end
end


%% ================================================================
% SAVE OUTPUT IMAGE (used only in 'infer' mode)
%% ================================================================
function saveOneOutput(fileEntry, status, config)
    try
        fullFilePath = fullfile(fileEntry.folder, fileEntry.name);
        [img, map] = imread(fullFilePath);
        if ~isempty(map)
            img = ind2rgb(img, map);
        end
        if ndims(img) == 2
            img = repmat(img, [1 1 3]);
        elseif size(img,3) == 4
            img = img(:,:,1:3);
        end
        if ~isa(img, 'uint8')
            img = im2uint8(img);
        end

        imgResized = imresize(img, config.TARGET_SIZE);

        if status == "PASS_GRADEABLE"
            labImg = rgb2lab(im2double(imgResized));
            L_channel = labImg(:,:,1) / 100;
            enhanced_L = adapthisteq(L_channel, 'ClipLimit', 0.02, ...
                'Distribution', 'rayleigh', 'NumTiles', [8 8]);
            labImg(:,:,1) = enhanced_L * 100;
            enhancedRGB = lab2rgb(labImg);
            enhancedRGB = im2uint8(enhancedRGB);
            for c = 1:3
                enhancedRGB(:,:,c) = medfilt2(enhancedRGB(:,:,c), [3 3]);
            end
            imwrite(enhancedRGB, fullfile(config.passDir, fileEntry.name));
        elseif status == "REVIEW"
            imwrite(imgResized, fullfile(config.reviewDir, fileEntry.name));
        else
            imwrite(imgResized, fullfile(config.rejectDir, fileEntry.name));
        end
    catch ME
        warning('Failed to save output for %s: %s', fileEntry.name, ME.message);
    end
end