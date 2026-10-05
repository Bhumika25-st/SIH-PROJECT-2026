%% Phase 3B: Probability Calibration + Uncertainty Estimation
% System: RETINA-AEGIS Module 3B
% Function: Calibrates raw softmax confidence via temperature scaling
%           (flowchart box 7), and computes a predictive-entropy
%           uncertainty score in [0,1] (flowchart box 8). Together these
%           feed the "confident enough?" decision gate (box 12).

%
% IMPORTANT - matches the CPU-optimized Phase 2 (v7 corrected):
%   - Reads pre-softmax logits from layer 'fc_dr' (NOT 'new_fc').
%   - Images must be normalized EXACTLY like cpuRead() in Phase 2
%     (resize -> im2single -> ImageNet mean/std), because Phase 2's
%     network has its input layer normalization set to 'none' - the
%     network expects already-normalized input, it will not normalize
%     raw pixels for you.
%   - Calibrates on imdsVal (NOT imdsTest) from Phase2_ValidationSplit.mat,
%     since imdsTest must stay untouched for unbiased final reporting.

clear; clc; close all;

%% 1. Configuration & Path Setup
scriptDir  = fileparts(mfilename('fullpath'));
modelFile  = fullfile(scriptDir, 'RetinaAegis_Classifier.mat');
splitFile  = fullfile(scriptDir, 'Phase2_ValidationSplit.mat');
outputDir  = fullfile(scriptDir, 'xai_reports', 'calibration_results');

if ~exist(outputDir, 'dir'), mkdir(outputDir); end

if ~exist(modelFile, 'file')
    error('Trained model not found: %s\nRun Phase 2 first.', modelFile);
end
if ~exist(splitFile, 'file')
    error('Validation split not found: %s\nRun Phase 2 first.', splitFile);
end

FC_LAYER_NAME = 'fc_dr';   % must match Phase 2's newFc layer name
inputSize     = [224 224]; % matches CFG.inputSize(1:2) in Phase 2

% ImageNet normalization constants - MUST match cpuRead() in Phase 2 exactly
meanRGB = single([0.485 0.456 0.406]);
stdRGB  = single([0.229 0.224 0.225]);

gradeNames = {'0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR'};
numClasses = 5;

% Decision-gate thresholds (flowchart box 12). These are heuristics you
% should tune once you see the calibration/uncertainty distributions on
% your own validation set - see the histograms this script produces.
CONFIDENCE_MIN_THRESHOLD = 0.60;  % below this calibrated confidence -> refer
UNCERTAINTY_MAX_THRESHOLD = 0.50; % above this normalized entropy -> refer

%% 2. Load Trained Network + Validation Split
fprintf('Loading trained classifier from %s ...\n', modelFile);
M = load(modelFile, 'trainedNet');
trainedNet = M.trainedNet;

fprintf('Loading validation split from %s ...\n', splitFile);
S = load(splitFile, 'valFiles', 'valLabels');
valFiles  = S.valFiles;
valLabels = S.valLabels;
numVal = numel(valFiles);
fprintf('Calibrating on %d validation images.\n', numVal);

%% 3. Extract Pre-Softmax Logits for the Validation Set
fprintf('Extracting logits from layer "%s" ...\n', FC_LAYER_NAME);
valLogits = zeros(numVal, numClasses, 'single');
valLabelsNum = double(valLabels) - 1; % categorical index -> grade 0-4

for i = 1:numVal
    img = readAndNormalize(valFiles{i}, inputSize, meanRGB, stdRGB);
    act = activations(trainedNet, img, FC_LAYER_NAME, 'OutputAs', 'rows');
    valLogits(i, :) = act;

    if mod(i, 50) == 0 || i == numVal
        fprintf('  ...%d / %d\n', i, numVal);
    end
end

%% 4. Baseline (Uncalibrated) Softmax Probabilities + Accuracy
rawProbs = softmaxRows(valLogits);
[~, rawPredIdx] = max(rawProbs, [], 2);
rawPred = rawPredIdx - 1;
rawAcc = mean(rawPred == valLabelsNum);
rawConf = max(rawProbs, [], 2);
rawECE = expectedCalibrationError(rawConf, rawPred == valLabelsNum, 10);

fprintf('\nUncalibrated (raw softmax): Accuracy = %.2f%% | ECE = %.4f\n', ...
    rawAcc*100, rawECE);

%% 5. Temperature Scaling
% Find the single scalar T > 0 that minimizes negative log-likelihood
% of the validation set when logits are divided by T before softmax.
% T > 1 softens (de-confidences) overconfident predictions; T < 1
% sharpens them. Network weights are NOT touched - this only rescales
% the confidence, never changes which class is predicted.
fprintf('\nFitting temperature scaling parameter...\n');
nllFun = @(T) negLogLikelihood(valLogits, valLabelsNum, T);
T_opt = fminbnd(nllFun, 0.05, 10);
fprintf('Optimal temperature T = %.4f\n', T_opt);

calibratedProbs = softmaxRows(valLogits / T_opt);
calibratedConf = max(calibratedProbs, [], 2);
[~, calibPredIdx] = max(calibratedProbs, [], 2);
calibPred = calibPredIdx - 1;
calibAcc = mean(calibPred == valLabelsNum); % should equal rawAcc - T doesn't change argmax
calibECE = expectedCalibrationError(calibratedConf, calibPred == valLabelsNum, 10);

fprintf('Calibrated (T=%.3f): Accuracy = %.2f%% | ECE = %.4f (lower is better)\n', ...
    T_opt, calibAcc*100, calibECE);

if calibECE >= rawECE
    warning(['Calibration did not reduce ECE (%.4f -> %.4f). This can happen ' ...
             'with small validation sets or an already well-calibrated model. ' ...
             'Inspect the reliability diagram before trusting T=%.3f.'], ...
             rawECE, calibECE, T_opt);
end

%% 6. Reliability Diagrams (Before vs After) - flowchart box 7 visual
figure('Name', 'Calibration: Before vs After', 'Position', [100 100 1000 450]);
subplot(1,2,1);
plotReliabilityDiagram(rawConf, rawPred == valLabelsNum, 10);
title(sprintf('Before Calibration (ECE=%.4f)', rawECE));

subplot(1,2,2);
plotReliabilityDiagram(calibratedConf, calibPred == valLabelsNum, 10);
title(sprintf('After Calibration (ECE=%.4f, T=%.3f)', calibECE, T_opt));

saveas(gcf, fullfile(outputDir, 'Calibration_Reliability_Diagrams.png'));

%% 7. Uncertainty Estimation (Predictive Entropy, normalized 0-1)
% Uses the CALIBRATED probabilities, since calibration is meant to run
% before uncertainty is read off the distribution (flowchart box 7 -> 8).
valEntropy = predictiveEntropy(calibratedProbs); % already normalized [0,1]

fprintf('\nUncertainty (normalized entropy) on validation set:\n');
fprintf('  Mean: %.3f | Median: %.3f | Max: %.3f\n', ...
    mean(valEntropy), median(valEntropy), max(valEntropy));

figure('Name', 'Uncertainty Distribution', 'Position', [150 150 700 450]);
histogram(valEntropy, 20);
xlabel('Normalized Predictive Entropy (0 = certain, 1 = maximally uncertain)');
ylabel('Number of Images');
title('Validation Set Uncertainty Distribution');
xline(UNCERTAINTY_MAX_THRESHOLD, 'r--', 'Refer threshold', 'LineWidth', 1.5);
saveas(gcf, fullfile(outputDir, 'Uncertainty_Distribution.png'));

% Sanity check: are high-uncertainty images actually the ones the model
% gets wrong more often? (they should be, if entropy is a useful signal)
wrongMask = calibPred ~= valLabelsNum;
fprintf('\nMean uncertainty on CORRECT predictions : %.3f\n', mean(valEntropy(~wrongMask)));
fprintf('Mean uncertainty on INCORRECT predictions: %.3f\n', mean(valEntropy(wrongMask)));
if mean(valEntropy(~wrongMask)) >= mean(valEntropy(wrongMask))
    warning(['Uncertainty is NOT higher on incorrect predictions than correct ' ...
             'ones. The entropy signal may not be discriminative on this ' ...
             'validation set - treat the refer-threshold decision with caution.']);
end

%% 8. Decision Gate Simulation (flowchart box 12) on Validation Set
% Shows what fraction of validation images would be auto-reported vs
% flagged for specialist review under the chosen thresholds, and how
% accurate the auto-reported subset is - this is the number that
% actually matters clinically (accuracy AFTER filtering by confidence).
autoProceedMask = (calibratedConf >= CONFIDENCE_MIN_THRESHOLD) & ...
                   (valEntropy <= UNCERTAINTY_MAX_THRESHOLD);

nAuto = sum(autoProceedMask);
nRefer = numVal - nAuto;
accAuto = mean(calibPred(autoProceedMask) == valLabelsNum(autoProceedMask));

fprintf('\nDecision Gate Simulation (thresholds: conf>=%.2f, entropy<=%.2f):\n', ...
    CONFIDENCE_MIN_THRESHOLD, UNCERTAINTY_MAX_THRESHOLD);
fprintf('  Auto-proceed : %d / %d (%.1f%%) | Accuracy on this subset: %.2f%%\n', ...
    nAuto, numVal, 100*nAuto/numVal, accAuto*100);
fprintf('  Refer to specialist: %d / %d (%.1f%%)\n', nRefer, numVal, 100*nRefer/numVal);

%% 9. Save Calibration + Uncertainty Parameters for Phase 5
calibrationParams = struct( ...
    'Temperature', T_opt, ...
    'FCLayerName', FC_LAYER_NAME, ...
    'InputSize', inputSize, ...
    'MeanRGB', meanRGB, ...
    'StdRGB', stdRGB, ...
    'ConfidenceMinThreshold', CONFIDENCE_MIN_THRESHOLD, ...
    'UncertaintyMaxThreshold', UNCERTAINTY_MAX_THRESHOLD, ...
    'ValidationECE_Raw', rawECE, ...
    'ValidationECE_Calibrated', calibECE, ...
    'ValidationAccuracy', calibAcc);

save(fullfile(outputDir, 'Phase3B_Calibration_Uncertainty.mat'), 'calibrationParams');
fprintf('\nCalibration + uncertainty parameters saved -> %s\n', ...
    fullfile(outputDir, 'Phase3B_Calibration_Uncertainty.mat'));
fprintf('Reliability diagrams and uncertainty histogram saved to: %s\n', outputDir);


%% ================================================================
% HELPER FUNCTIONS
%% ================================================================

function img = readAndNormalize(filePath, targetSize, meanRGB, stdRGB)
    % Mirrors cpuRead() from Phase 2 EXACTLY (minus augmentation, which
    % should never be applied at inference/calibration time).
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

function probs = softmaxRows(logits)
    % Numerically stable row-wise softmax
    z = logits - max(logits, [], 2);
    ez = exp(z);
    probs = ez ./ sum(ez, 2);
end

function nll = negLogLikelihood(logits, trueLabelsNum, T)
    probs = softmaxRows(double(logits) / T);
    idx = sub2ind(size(probs), (1:size(probs,1))', trueLabelsNum + 1);
    p_true = probs(idx);
    p_true = max(p_true, eps); % avoid log(0)
    nll = -mean(log(p_true));
end

function ece = expectedCalibrationError(confidences, correctMask, numBins)
    % Standard ECE: bins predictions by confidence, compares average
    % confidence to actual accuracy within each bin.
    edges = linspace(0, 1, numBins + 1);
    ece = 0;
    n = numel(confidences);
    for b = 1:numBins
        inBin = confidences > edges(b) & confidences <= edges(b+1);
        if any(inBin)
            binConf = mean(confidences(inBin));
            binAcc  = mean(correctMask(inBin));
            ece = ece + (sum(inBin)/n) * abs(binAcc - binConf);
        end
    end
end

function plotReliabilityDiagram(confidences, correctMask, numBins)
    edges = linspace(0, 1, numBins + 1);
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;
    binAcc = nan(1, numBins);
    for b = 1:numBins
        inBin = confidences > edges(b) & confidences <= edges(b+1);
        if any(inBin)
            binAcc(b) = mean(correctMask(inBin));
        end
    end
    bar(binCenters, binAcc, 0.9, 'FaceColor', [0.2 0.5 0.8]); hold on;
    plot([0 1], [0 1], 'k--', 'LineWidth', 1.5);
    xlabel('Predicted Confidence'); ylabel('Observed Accuracy');
    xlim([0 1]); ylim([0 1]); grid on;
end

function H = predictiveEntropy(probs)
    % Normalized predictive entropy in [0, 1]. 0 = fully certain
    % (one-hot), 1 = maximally uncertain (uniform over all classes).
    numClasses = size(probs, 2);
    p = max(probs, eps); % avoid log(0)
    rawEntropy = -sum(p .* log(p), 2);
    H = rawEntropy / log(numClasses); % normalize by max possible entropy
end