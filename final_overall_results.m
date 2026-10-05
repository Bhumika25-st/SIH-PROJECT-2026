%% Phase 5: Final Explainable Output (Combined Report)
% System: RETINA-AEGIS Module 5
% Function: Fuses the three parallel XAI branches - Grad-CAM (Phase 3),
%           Calibration + Uncertainty (Phase 3B), and Lesion Detection
%           (Phase 4) - with Phase 2's OFFICIAL model output (TTA-averaged
%           predictions, the ones Phase 2 actually reported 78.73%
%           accuracy / 82.91% referable sensitivity for) into one final
%           per-patient explainable panel and a master CSV. Implements
%           the decision gate (flowchart box 12) for real and the final
%           output format (flowchart box 13).

%
% IMPORTANT - WHY THIS RE-RUNS TTA INSTEAD OF REUSING PHASE 3's PREDICTIONS:
%   Phase 3 (Grad-CAM) used a single plain predict() call per image, NOT
%   TTA - that's why its own sanity-check accuracy (77.73%) was ~1pp below
%   Phase 2's officially reported 78.73%. If Phase 5 just reused Phase 3's
%   single-pass predictions as the "final" decision, the report would be
%   built on a WEAKER, non-official prediction than the one Phase 2 was
%   actually validated on - and for any image where single-pass and TTA
%   disagree, the Grad-CAM heatmap would silently be explaining a
%   DIFFERENT class than the one reported as the final decision. So this
%   script loads the trained model directly and recomputes TTA-averaged
%   predictions itself (the same ~2-3 minute cost Phase 2 already paid
%   once), uses THAT as the official prediction, and explicitly flags any
%   row where Grad-CAM's explained class disagrees with it.
%
% REQUIRES (all must have been run already):
%   - Phase 2  -> RetinaAegis_Classifier.mat, Phase2_ValidationSplit.mat
%   - Phase 3  -> xai_reports/gradcam_results/Phase3_GradCAM_Results.mat
%   - Phase 3B -> xai_reports/calibration_results/Phase3B_Calibration_Uncertainty.mat
%   - Phase 4  -> xai_reports/lesion_results/Phase4_Lesion_Results.mat
%
% CALIBRATION NOTE: for the single-pass probabilities this script would
% otherwise use, softmax(log(p_raw)/T) == softmax(logits/T) EXACTLY
% (softmax is invariant to constant shifts, and log(p_raw) is such a
% shift of the true logits). For TTA-AVERAGED probabilities specifically,
% this is a reasonable, monotonic APPROXIMATION rather than an exact
% equivalence, since an average of 5 softmax outputs is not itself the
% softmax of one single logit vector. It still correctly sharpens/softens
% confidence in the right direction - just noting it's not mathematically
% exact in this specific TTA case, unlike the single-pass case.

clear; clc; close all;

%% 1. Configuration & Path Setup
scriptDir = fileparts(mfilename('fullpath'));
modelFile = fullfile(scriptDir, 'RetinaAegis_Classifier.mat');
splitFile = fullfile(scriptDir, 'Phase2_ValidationSplit.mat');

xaiDir        = fullfile(scriptDir, 'xai_reports');
gradcamFile   = fullfile(xaiDir, 'gradcam_results', 'Phase3_GradCAM_Results.mat');
calibFile     = fullfile(xaiDir, 'calibration_results', 'Phase3B_Calibration_Uncertainty.mat');
lesionFile    = fullfile(xaiDir, 'lesion_results', 'Phase4_Lesion_Results.mat');

% NOTE: outputDir now written at scriptDir (main folder) level, not
% nested under xai_reports - change this back to fullfile(xaiDir,'final_results')
% if you'd rather keep it nested.
outputDir = fullfile(scriptDir, 'xai_reports','final_results');
panelDir  = fullfile(outputDir, 'panels');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
if ~exist(panelDir, 'dir'),  mkdir(panelDir);  end

requiredFiles = {modelFile, splitFile, gradcamFile, calibFile, lesionFile};
requiredNames = {'RetinaAegis_Classifier.mat', 'Phase2_ValidationSplit.mat', ...
    'Phase3_GradCAM_Results.mat', 'Phase3B_Calibration_Uncertainty.mat', 'Phase4_Lesion_Results.mat'};
for i = 1:numel(requiredFiles)
    if ~exist(requiredFiles{i}, 'file')
        error(['Missing required file: %s\n' ...
               'Run all of Phase 2, 3, 3B, and 4 before Phase 5.'], requiredNames{i});
    end
end

gradeNames = {'0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR'};
DISPLAY_SIZE = [260 260]; % common size for the 3-panel image montage
REFERABLE_THRESHOLD = 2;  % Grade 2+ = clinically referable, per PRD

meanRGB = single([0.485 0.456 0.406]);
stdRGB  = single([0.229 0.224 0.225]);

%% 2. Load Everything
fprintf('Loading Phase 2 trained model + config...\n');
M = load(modelFile, 'trainedNet', 'CFG', 'acc', 'referableSens', 'referableSpec');
trainedNet = M.trainedNet;
CFG = M.CFG;
fprintf('  Phase 2 officially reported: Accuracy=%.2f%% | Referable Sens=%.2f%% | Referable Spec=%.2f%%\n', ...
    M.acc*100, M.referableSens*100, M.referableSpec*100);

fprintf('Loading Phase 2 test split...\n');
S = load(splitFile, 'testFiles', 'testLabels');
testFiles = S.testFiles;
testLabels = S.testLabels;
numImages = numel(testFiles);

fprintf('Loading Phase 3 Grad-CAM results (for heatmaps + explained class)...\n');
G = load(gradcamFile, 'gradcamResults');
gradcamResults = G.gradcamResults;

fprintf('Loading Phase 3B calibration parameters...\n');
C = load(calibFile, 'calibrationParams');
calibParams = C.calibrationParams;
T_opt = calibParams.Temperature;
confThresh = calibParams.ConfidenceMinThreshold;
uncThresh  = calibParams.UncertaintyMaxThreshold;
fprintf('  Temperature=%.3f | ConfThresh=%.2f | UncThresh=%.2f\n', T_opt, confThresh, uncThresh);

fprintf('Loading Phase 4 lesion results...\n');
L = load(lesionFile, 'lesionResults');
lesionResults = L.lesionResults;

% Index Grad-CAM and lesion results by filename for robust lookup
gradcamMap = containers.Map();
for i = 1:numel(gradcamResults)
    gradcamMap(char(gradcamResults(i).Filename)) = i;
end
lesionMap = containers.Map();
for i = 1:numel(lesionResults)
    lesionMap(char(lesionResults(i).Filename)) = i;
end

%% 3. Recompute OFFICIAL TTA-Averaged Predictions (Phase 2's real output)
fprintf('\nRecomputing TTA-averaged predictions (this is Phase 2''s OFFICIAL output)...\n');
ttaProbs = ttaScoreAveraging(trainedNet, testFiles, CFG, meanRGB, stdRGB);
[~, ttaPredIdxOneBased] = max(ttaProbs, [], 2);
ttaPredIdx = ttaPredIdxOneBased - 1;
testLabelsNum = double(testLabels) - 1;

ttaAcc = mean(ttaPredIdx == testLabelsNum);
fprintf('TTA re-scoring accuracy: %.2f%% (sanity check vs Phase 2''s reported %.2f%%)\n', ...
    ttaAcc*100, M.acc*100);
if abs(ttaAcc - M.acc) > 0.03
    warning(['TTA re-scoring accuracy differs from Phase 2''s reported accuracy by more ' ...
             'than 3pp (%.2f%% vs %.2f%%). Check that testFiles/testLabels and the model ' ...
             'match exactly what Phase 2 evaluated.'], ttaAcc*100, M.acc*100);
end

%% 4. Results Container
finalResults = struct('Filename', {}, 'TrueGrade', {}, 'PredictedGrade', {}, ...
    'RawConfidence', {}, 'CalibratedConfidence', {}, 'Uncertainty', {}, 'UncertaintyStatus', {}, ...
    'IsReferableDR', {}, 'Decision', {}, 'GradCAMClassMismatch', {}, ...
    'NumMicroaneurysms', {}, 'NumHemorrhages', {}, 'NumHardExudates', {}, ...
    'NumSoftExudates', {}, 'Prob_Grade0', {}, 'Prob_Grade1', {}, 'Prob_Grade2', {}, ...
    'Prob_Grade3', {}, 'Prob_Grade4', {}, 'PanelFile', {}, 'FullImagePath', {});

% Expanded schema: adds GradCAMClassMismatch (was silently missing from
% the previous CSV output), full per-lesion breakdown alongside the
% combined totals, full 5-class calibrated probability distribution,
% an explicit UncertaintyStatus column, PanelFile for traceability to
% the panel image, and FullImagePath back to the source image.
reportTable = table('Size', [0 24], ...
    'VariableTypes', {'string','string','string','double','double','double','string', ...
                       'logical','string','logical', ...
                       'double','double','double','double', ...
                       'double','double','double', ...
                       'double','double','double','double','double', ...
                       'string','string'}, ...
    'VariableNames', {'Filename','TrueGrade','PredictedGrade','RawConfidence', ...
                       'CalibratedConfidence','Uncertainty','UncertaintyStatus', ...
                       'IsReferableDR','Decision','GradCAMClassMismatch', ...
                       'NumMicroaneurysms','NumHemorrhages','NumHardExudates','NumSoftExudates', ...
                       'TotalMA_Hem','TotalExudates','LesionBurden', ...
                       'Prob_Grade0','Prob_Grade1','Prob_Grade2','Prob_Grade3','Prob_Grade4', ...
                       'PanelFile','FullImagePath'});

%% 5. Main Fusion Loop
for k = 1:numImages
    filePath = testFiles{k};
    [~, baseName, ext] = fileparts(filePath);
    baseNameFull = [baseName ext];
    trueIdx = testLabelsNum(k);

    %% Step A: OFFICIAL Prediction + Calibrated Confidence (TTA-based)
    rawProbsTTA = max(ttaProbs(k, :), eps);
    calibratedProbs = softmaxRows(log(rawProbsTTA) / T_opt);
    calibConf = max(calibratedProbs);
    uncertainty = predictiveEntropy(calibratedProbs);
    predictedIdx = ttaPredIdx(k); % OFFICIAL class, matches Phase 2's reported metrics
    isReferableDR = predictedIdx >= REFERABLE_THRESHOLD;

    if uncertainty > 0.5
        uncStatus = "High";
    else
        uncStatus = "Low";
    end

    %% Step B: Decision Gate (flowchart box 12)
    if calibConf < confThresh || uncertainty > uncThresh
        decision = "REFER: Low Confidence / High Uncertainty - Specialist Review Required";
    elseif isReferableDR
        decision = "REFERABLE DR - Refer to Ophthalmologist";
    else
        decision = "Routine / Non-Referable - Continue Regular Screening";
    end

    %% Step C: Pull Matching Grad-CAM + Lesion Data, Flag Class Mismatch
    gradCAMMismatch = false;
    if isKey(gradcamMap, baseNameFull)
        gr = gradcamResults(gradcamMap(baseNameFull));
        gradCAMExplainedIdx = gr.PredictedIdx;
        if gradCAMExplainedIdx ~= predictedIdx
            gradCAMMismatch = true;
        end
        gradcamOverlayFile = char(gr.OverlayFile);
    else
        warning('No Grad-CAM result found for %s.', baseNameFull);
        gradcamOverlayFile = '';
    end

    if isKey(lesionMap, baseNameFull)
        les = lesionResults(lesionMap(baseNameFull));
    else
        warning('No lesion result found for %s - using zero counts.', baseNameFull);
        les = struct('NumMicroaneurysms', 0, 'NumHemorrhages', 0, ...
                      'NumHardExudates', 0, 'NumSoftExudates', 0, 'OverlayFile', "");
    end

    %% Step D: Build the Final Explainable Panel (flowchart box 13)
    originalImg = imresize(imread(filePath), DISPLAY_SIZE);
    if ~isempty(gradcamOverlayFile)
        gradcamOverlay = imresize(imread(gradcamOverlayFile), DISPLAY_SIZE);
    else
        gradcamOverlay = originalImg;
    end
    if isfield(les, 'OverlayFile') && strlength(string(les.OverlayFile)) > 0
        lesionOverlay = imresize(imread(char(les.OverlayFile)), DISPLAY_SIZE);
    else
        lesionOverlay = originalImg;
    end

    panelFile = fullfile(panelDir, [baseName '_final.png']);
    buildFinalPanel(originalImg, gradcamOverlay, lesionOverlay, ...
        gradeNames{predictedIdx+1}, gradeNames{trueIdx+1}, calibConf, uncertainty, ...
        decision, les, gradCAMMismatch, panelFile);

    %% Step E: Log
    finalResults(end+1) = struct( ...
        'Filename', string(baseNameFull), ...
        'TrueGrade', string(gradeNames{trueIdx+1}), ...
        'PredictedGrade', string(gradeNames{predictedIdx+1}), ...
        'RawConfidence', max(rawProbsTTA)*100, ...
        'CalibratedConfidence', calibConf*100, ...
        'Uncertainty', uncertainty, ...
        'UncertaintyStatus', uncStatus, ...
        'IsReferableDR', isReferableDR, ...
        'Decision', decision, ...
        'GradCAMClassMismatch', gradCAMMismatch, ...
        'NumMicroaneurysms', les.NumMicroaneurysms, ...
        'NumHemorrhages', les.NumHemorrhages, ...
        'NumHardExudates', les.NumHardExudates, ...
        'NumSoftExudates', les.NumSoftExudates, ...
        'Prob_Grade0', calibratedProbs(1)*100, ...
        'Prob_Grade1', calibratedProbs(2)*100, ...
        'Prob_Grade2', calibratedProbs(3)*100, ...
        'Prob_Grade3', calibratedProbs(4)*100, ...
        'Prob_Grade4', calibratedProbs(5)*100, ...
        'PanelFile', string(panelFile), ...
        'FullImagePath', string(filePath)); %#ok<SAGROW>

    reportTable = [reportTable; {string(baseNameFull), string(gradeNames{trueIdx+1}), ...
        string(gradeNames{predictedIdx+1}), max(rawProbsTTA)*100, calibConf*100, uncertainty, uncStatus, ...
        isReferableDR, decision, gradCAMMismatch, ...
        les.NumMicroaneurysms, les.NumHemorrhages, les.NumHardExudates, les.NumSoftExudates, ...
        les.NumMicroaneurysms + les.NumHemorrhages, ...
        les.NumHardExudates + les.NumSoftExudates, ...
        les.NumMicroaneurysms + les.NumHemorrhages + les.NumHardExudates + les.NumSoftExudates, ...
        calibratedProbs(1)*100, calibratedProbs(2)*100, calibratedProbs(3)*100, ...
        calibratedProbs(4)*100, calibratedProbs(5)*100, ...
        string(panelFile), string(filePath)}]; %#ok<AGROW>

    if mod(k, 25) == 0 || k == numImages
        fprintf('Processed %d / %d images...\n', k, numImages);
    end
end

%% 6. Persist Final Results
save(fullfile(outputDir, 'Phase5_Final_Results.mat'), 'finalResults');
writetable(reportTable, fullfile(outputDir, 'Phase5_Final_Report.csv'));

%% 7. Summary Statistics
nTotal = height(reportTable);
nReferOnUncertainty = sum(startsWith(reportTable.Decision, "REFER: Low"));
nRoutine = sum(startsWith(reportTable.Decision, "Routine"));
nMismatch = sum(reportTable.GradCAMClassMismatch);

autoReportMask = ~startsWith(reportTable.Decision, "REFER: Low");
autoReportAcc = mean(reportTable.TrueGrade(autoReportMask) == reportTable.PredictedGrade(autoReportMask));

fprintf('\n============================================\n');
fprintf(' PHASE 5 FINAL REPORT SUMMARY (n=%d)\n', nTotal);
fprintf('============================================\n');
fprintf('Official prediction source: TTA-averaged (matches Phase 2''s reported %.2f%% accuracy)\n', M.acc*100);
fprintf('Referred - Low Confidence/High Uncertainty : %d (%.1f%%)\n', nReferOnUncertainty, 100*nReferOnUncertainty/nTotal);
fprintf('Referable DR (Grade 2+), auto-confident     : %d (%.1f%%)\n', ...
    sum(reportTable.IsReferableDR & autoReportMask), 100*sum(reportTable.IsReferableDR & autoReportMask)/nTotal);
fprintf('Routine / Non-Referable, auto-confident      : %d (%.1f%%)\n', nRoutine, 100*nRoutine/nTotal);
fprintf('Accuracy on auto-reported subset             : %.2f%%\n', autoReportAcc*100);
fprintf('Grad-CAM/Official-prediction class mismatches: %d (%.1f%%)\n', nMismatch, 100*nMismatch/nTotal);
if nMismatch > 0
    fprintf(['  -> For these images, the Grad-CAM heatmap explains a DIFFERENT class\n' ...
             '     than the official TTA-based decision. Flagged in the CSV/panels -\n' ...
             '     worth manually reviewing before using in a demo.\n']);
end
fprintf('\nOutputs:\n');
fprintf('  Per-patient panels : %s\n', panelDir);
fprintf('  Master CSV         : %s\n', fullfile(outputDir, 'Phase5_Final_Report.csv'));
fprintf('  Results struct     : %s\n', fullfile(outputDir, 'Phase5_Final_Results.mat'));


%% ================================================================
% HELPER FUNCTIONS
%% ================================================================

function scoreAcc = ttaScoreAveraging(net, files, CFG, meanRGB, stdRGB)
    % Same TTA logic as cpuTTA_fixed.m, but returns the full averaged
    % probability matrix (numFiles x numClasses) instead of just the
    % argmax label, since Phase 5 needs the probabilities themselves
    % for calibration and uncertainty, not just the predicted class.
    nFiles = numel(files);
    numClasses = 5;
    scoreAcc = zeros(nFiles, numClasses, 'single');
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
            img = imread(files{i});
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

        scores = predict(net, batchImgs, 'ExecutionEnvironment', 'cpu');

        for i = 1:nChunk
            s = (i-1)*CFG.numTTA + 1;
            e = i*CFG.numTTA;
            scoreAcc(iStart+i-1, :) = mean(scores(s:e,:), 1);
        end

        fprintf('  TTA chunk %d/%d (%.1fs)\n', c, numChunks, toc);
    end
end

function probs = softmaxRows(logits)
    z = logits - max(logits, [], 2);
    ez = exp(z);
    probs = ez ./ sum(ez, 2);
end

function H = predictiveEntropy(probs)
    numClasses = numel(probs);
    p = max(probs, eps);
    rawEntropy = -sum(p .* log(p));
    H = rawEntropy / log(numClasses);
end

function buildFinalPanel(originalImg, gradcamOverlay, lesionOverlay, predGrade, ...
    trueGrade, calibConf, uncertainty, decision, les, gradCAMMismatch, outFile)

    fig = figure('Visible', 'off', 'Position', [100 100 900 440], 'Color', 'w');
    tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile; imshow(originalImg); title('Original (Enhanced)', 'FontSize', 10);
    nexttile; imshow(gradcamOverlay);
    if gradCAMMismatch
        title('Grad-CAM (\color{red}different class!)', 'FontSize', 10);
    else
        title('Grad-CAM', 'FontSize', 10);
    end
    nexttile; imshow(lesionOverlay); title('Lesion Overlay', 'FontSize', 10);

    if uncertainty > 0.5
        uncStatus = 'High';
    else
        uncStatus = 'Low';
    end

    mismatchNote = '';
    if gradCAMMismatch
        mismatchNote = sprintf('\n(NOTE: Grad-CAM heatmap explains a different class than this official prediction)');
    end

    summaryText = sprintf(['Predicted (official, TTA): %s   |   True (ref. label): %s\n' ...
        'Calibrated Confidence: %.1f%%   |   Uncertainty: %.2f (%s)\n' ...
        'Lesions - MA: %d | Hem: %d | Hard Exudates: %d | Soft Exudates: %d\n' ...
        'DECISION: %s%s'], ...
        predGrade, trueGrade, calibConf*100, uncertainty, uncStatus, ...
        les.NumMicroaneurysms, les.NumHemorrhages, les.NumHardExudates, les.NumSoftExudates, ...
        decision, mismatchNote);

    sgtitle(summaryText, 'FontSize', 10, 'FontWeight', 'normal', 'Interpreter', 'none');

    exportgraphics(fig, outFile, 'Resolution', 130);
    close(fig);
end