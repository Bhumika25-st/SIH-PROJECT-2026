%% Phase 4: Lesion Detection & Segmentation (Classical Computer Vision)
% System: RETINA-AEGIS Module 4
% Function: Detects and segments retinal lesions - microaneurysms,
%           hemorrhages, hard exudates, soft exudates - using classical
%           image processing (NO deep learning), and produces a
%           color-coded overlay matching the flowchart legend
%           (flowchart boxes 10 + 11).
%
% IMPORTANT NOTES:
%   - This is a heuristic, threshold/morphology-based pipeline, not a
%     clinically validated lesion detector. It is intended to produce
%     plausible visual evidence to accompany Grad-CAM in the final
%     explainable report (Phase 5), not a diagnostic-grade segmentation.
%     Treat lesion COUNTS as rough indicators, not ground truth.
%   - Runs on the SAME imdsTest population as Phase 3 (Grad-CAM), loaded
%     from Phase2_ValidationSplit.mat, so all XAI outputs describe the
%     same set of images.
%   - Operates on the Phase 1 CLAHE-enhanced images at their NATIVE
%     resolution (NOT resized to 224x224) - lesion detail (especially
%     microaneurysms, which are only a few pixels wide) is lost at
%     network input resolution, so this reads files directly rather
%     than reusing Phase 3's 224x224 normalized tensors.

clear; clc; close all;

%% 1. Configuration & Path Setup
scriptDir = fileparts(mfilename('fullpath'));
splitFile = fullfile(scriptDir, 'Phase2_ValidationSplit.mat');
outputDir = fullfile(scriptDir, 'xai_reports', 'lesion_results');
overlayDir = fullfile(outputDir, 'overlays');
maskDir    = fullfile(outputDir, 'masks');

if ~exist(outputDir, 'dir'),  mkdir(outputDir);  end
if ~exist(overlayDir, 'dir'), mkdir(overlayDir); end
if ~exist(maskDir, 'dir'),    mkdir(maskDir);    end

if ~exist(splitFile, 'file')
    error('Split file not found: %s\nRun Phase 2 first.', splitFile);
end

fprintf('Loading test split from %s ...\n', splitFile);
S = load(splitFile, 'testFiles', 'testLabels');
testFiles = S.testFiles;
numImages = numel(testFiles);
fprintf('Running lesion detection on %d test-set images.\n', numImages);

% Color legend, matching the flowchart's Segmentation Overlay box exactly
LESION_COLORS = struct( ...
    'Microaneurysm', [0   0   255], ...   % Blue
    'Hemorrhage',    [255 0   0],   ...   % Red
    'HardExudate',   [255 255 0],   ...   % Yellow
    'SoftExudate',   [0   200 0]);        % Green

OVERLAY_ALPHA = 0.55;

%% 2. Results Container
lesionResults = struct('Filename', {}, 'NumMicroaneurysms', {}, 'NumHemorrhages', {}, ...
    'NumHardExudates', {}, 'NumSoftExudates', {}, 'MaskFile', {}, 'OverlayFile', {});

reportTable = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Filename', 'Microaneurysms', 'Hemorrhages', 'HardExudates', 'SoftExudates'});

%% 3. Main Loop
for k = 1:numImages
    filePath = testFiles{k};
    [~, baseName, ext] = fileparts(filePath);

    img = imread(filePath);
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    end

    grayImg  = rgb2gray(img);
    greenCh  = img(:,:,2); % highest lesion contrast channel in fundus photography

    %% Step A: Field-of-View (FOV) Mask - exclude black background
    fovMask = grayImg > 15;
    fovMask = imfill(fovMask, 'holes');
    cc = bwconncomp(fovMask);
    if cc.NumObjects > 0
        areas = cellfun(@numel, cc.PixelIdxList);
        [~, biggest] = max(areas);
        fovMask = false(size(fovMask));
        fovMask(cc.PixelIdxList{biggest}) = true;
    end
    fovMask = imerode(fovMask, strel('disk', 5)); % pull in from the rim, avoids edge artifacts

    %% Step B: Optic Disc Detection (exclusion zone)
    % The optic disc is naturally bright and yellowish - without
    % excluding it, hard exudate detection would flag it as a false
    % positive lesion on every single image.
    opticDiscMask = detectOpticDisc(grayImg, fovMask);

    % Working mask: FOV minus optic disc, for all lesion searches
    searchMask = fovMask & ~opticDiscMask;

    %% Step C: Microaneurysms - small dark round blobs
    maMask = detectMicroaneurysms(greenCh, searchMask);

    %% Step D: Hemorrhages - larger dark irregular blobs
    hemMask = detectHemorrhages(greenCh, searchMask);
    % A blob can't be both a hemorrhage and a microaneurysm - resolve by
    % size (already mostly separated by structuring element size, but
    % clean up any overlap by removing hemorrhage pixels from the MA mask)
    maMask = maMask & ~hemMask;

    %% Step E: Hard Exudates - bright yellow-white spots
    heMask = detectHardExudates(img, searchMask);

    %% Step F: Soft Exudates (cotton wool spots) - fluffy, lower-contrast bright patches
    seMask = detectSoftExudates(greenCh, searchMask & ~heMask);

    %% Step G: Count Lesions (connected components per class)
    numMA  = countBlobs(maMask);
    numHem = countBlobs(hemMask);
    numHE  = countBlobs(heMask);
    numSE  = countBlobs(seMask);

    %% Step H: Build Color-Coded Overlay (flowchart box 11)
    overlayImg = buildLesionOverlay(img, maMask, hemMask, heMask, seMask, LESION_COLORS, OVERLAY_ALPHA);
    overlayFile = fullfile(overlayDir, [baseName '_lesions.png']);
    imwrite(overlayImg, overlayFile);

    %% Step I: Save Raw Masks (for Phase 5 fusion with Grad-CAM heatmaps)
    maskFile = fullfile(maskDir, [baseName '_masks.mat']);
    save(maskFile, 'maMask', 'hemMask', 'heMask', 'seMask', 'opticDiscMask', 'fovMask');

    %% Step J: Log
    lesionResults(end+1) = struct( ...
        'Filename', string(baseName) + string(ext), ...
        'NumMicroaneurysms', numMA, 'NumHemorrhages', numHem, ...
        'NumHardExudates', numHE, 'NumSoftExudates', numSE, ...
        'MaskFile', string(maskFile), 'OverlayFile', string(overlayFile)); %#ok<SAGROW>

    reportTable = [reportTable; {string(baseName) + string(ext), numMA, numHem, numHE, numSE}]; %#ok<AGROW>

    if mod(k, 25) == 0 || k == numImages
        fprintf('Processed %d / %d images...\n', k, numImages);
    end
end

%% 4. Persist Results for Phase 5
save(fullfile(outputDir, 'Phase4_Lesion_Results.mat'), 'lesionResults');
writetable(reportTable, fullfile(outputDir, 'Phase4_Lesion_Report.csv'));

fprintf('\nExecution Completed:\n');
fprintf('  Images Processed: %d\n', numImages);
fprintf('  Mean lesions/image -> MA: %.1f | Hem: %.1f | HE: %.1f | SE: %.1f\n', ...
    mean(reportTable.Microaneurysms), mean(reportTable.Hemorrhages), ...
    mean(reportTable.HardExudates), mean(reportTable.SoftExudates));
fprintf('  Overlays saved to: %s\n', overlayDir);
fprintf('  Raw masks saved to: %s\n', maskDir);
fprintf('  Results struct saved to: %s\n', fullfile(outputDir, 'Phase4_Lesion_Results.mat'));


%% ================================================================
% HELPER FUNCTIONS
%% ================================================================

function opticDiscMask = detectOpticDisc(grayImg, fovMask)
    % Optic disc = the single brightest large connected region within
    % the retina. Threshold at the top ~2% of intensity within the FOV,
    % keep the largest resulting blob, then dilate generously (the disc
    % has a soft/glowing edge, not a hard boundary).
    pixelsInFOV = grayImg(fovMask);
    if isempty(pixelsInFOV)
        opticDiscMask = false(size(grayImg));
        return;
    end
    brightThresh = prctile(double(pixelsInFOV), 98);
    candidateMask = (grayImg >= brightThresh) & fovMask;
    candidateMask = bwareaopen(candidateMask, 30);

    cc = bwconncomp(candidateMask);
    if cc.NumObjects == 0
        opticDiscMask = false(size(grayImg));
        return;
    end
    areas = cellfun(@numel, cc.PixelIdxList);
    [~, biggest] = max(areas);
    opticDiscMask = false(size(grayImg));
    opticDiscMask(cc.PixelIdxList{biggest}) = true;
    opticDiscMask = imdilate(opticDiscMask, strel('disk', 12));
end

function maMask = detectMicroaneurysms(greenCh, searchMask)
    % Microaneurysms: tiny (a few pixels), round, DARKER than local
    % background. Black top-hat (complement top-hat) with a small disk
    % isolates small dark structures against a locally bright surround.
    complementImg = imcomplement(greenCh);
    se = strel('disk', 3);
    tophat = imtophat(complementImg, se);
    tophat(~searchMask) = 0;

    thresh = mean(tophat(searchMask)) + 2.5*std(double(tophat(searchMask)));
    maMask = tophat > thresh;

    % Keep only small, roughly round blobs (rules out thin vessel
    % fragments that survive the top-hat, which are elongated)
    maMask = bwareaopen(maMask, 2);
    maMask = filterByShape(maMask, [2 40], 0.5); % area range + min circularity
end

function hemMask = detectHemorrhages(greenCh, searchMask)
    % Hemorrhages: larger and more irregular than microaneurysms, still
    % darker than surrounding tissue. Same top-hat idea with a bigger
    % structuring element to catch larger dark blobs, looser circularity
    % requirement since hemorrhages are often irregular/flame-shaped.
    complementImg = imcomplement(greenCh);
    se = strel('disk', 8);
    tophat = imtophat(complementImg, se);
    tophat(~searchMask) = 0;

    thresh = mean(tophat(searchMask)) + 2.0*std(double(tophat(searchMask)));
    hemMask = tophat > thresh;

    hemMask = bwareaopen(hemMask, 25);
    hemMask = filterByShape(hemMask, [25 800], 0.15);
end

function heMask = detectHardExudates(rgbImg, searchMask)
    % Hard exudates: bright, sharply-defined, yellowish-white spots.
    % Convert to Lab; exudates are high-L with a positive (yellow) b*.
    labImg = rgb2lab(im2double(rgbImg));
    Lch = labImg(:,:,1);
    bch = labImg(:,:,3); % b* channel: positive = yellow, negative = blue

    brightMask = Lch > (mean(Lch(searchMask)) + 2.0*std(Lch(searchMask)));
    yellowMask = bch > (mean(bch(searchMask)) + 0.5*std(bch(searchMask)));

    heMask = brightMask & yellowMask & searchMask;
    heMask = bwareaopen(heMask, 8);
    heMask = filterByShape(heMask, [8 2000], 0.2);
end

function seMask = detectSoftExudates(greenCh, searchMask)
    % Soft exudates (cotton wool spots): larger, fluffy, LOWER-contrast
    % bright patches (unlike hard exudates, edges are blurred, not
    % sharp). White top-hat with a large structuring element picks up
    % broad bright regions; a moderate (not extreme) threshold avoids
    % re-flagging the sharp hard-exudate spots already excluded upstream.
    se = strel('disk', 15);
    tophat = imtophat(greenCh, se);
    tophat(~searchMask) = 0;

    thresh = mean(tophat(searchMask)) + 1.5*std(double(tophat(searchMask)));
    seMask = tophat > thresh;

    seMask = bwareaopen(seMask, 100);
    seMask = filterByShape(seMask, [100 5000], 0.1);
end

function filteredMask = filterByShape(mask, areaRange, minCircularity)
    % Keeps only connected components within an area range AND above a
    % minimum circularity (4*pi*Area/Perimeter^2; 1.0 = perfect circle).
    % Filters out thin vessel fragments and other non-lesion artifacts
    % that survive the intensity thresholds above.
    cc = bwconncomp(mask);
    stats = regionprops(cc, 'Area', 'Perimeter');
    filteredMask = false(size(mask));

    for i = 1:cc.NumObjects
        area = stats(i).Area;
        perim = max(stats(i).Perimeter, 1);
        circularity = 4*pi*area / (perim^2);
        if area >= areaRange(1) && area <= areaRange(2) && circularity >= minCircularity
            filteredMask(cc.PixelIdxList{i}) = true;
        end
    end
end

function n = countBlobs(mask)
    cc = bwconncomp(mask);
    n = cc.NumObjects;
end

function overlayImg = buildLesionOverlay(originalImg, maMask, hemMask, heMask, seMask, colors, alpha)
    overlayImg = double(originalImg);
    overlayImg = applyColorMask(overlayImg, maMask,  colors.Microaneurysm, alpha);
    overlayImg = applyColorMask(overlayImg, hemMask, colors.Hemorrhage,    alpha);
    overlayImg = applyColorMask(overlayImg, heMask,  colors.HardExudate,  alpha);
    overlayImg = applyColorMask(overlayImg, seMask,  colors.SoftExudate,  alpha);
    overlayImg = uint8(overlayImg);
end

function img = applyColorMask(img, mask, color, alpha)
    if ~any(mask(:))
        return;
    end
    for c = 1:3
        channel = img(:,:,c);
        channel(mask) = (1-alpha)*channel(mask) + alpha*color(c);
        img(:,:,c) = channel;
    end
end