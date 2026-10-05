function predLabels = cpuTTA_fixed(net, imds, CFG, meanRGB, stdRGB)
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
        scores = predict(net, batchImgs, 'ExecutionEnvironment', 'cpu');
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