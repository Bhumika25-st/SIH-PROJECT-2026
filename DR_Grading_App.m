function DR_Grading_App
% RETINA-AEGIS - Accessible DR Grading UI
% Simple, reliable, eye-pleasing for all ages
% Run: >> DR_Grading_App

    %% ---- LOAD MODEL ----
    modelFiles = {
        'RetinaAegis_CPU_v8_Grade3Focus.mat', 'trainedNet_v8'
        'RetinaAegis_FULL_Results.mat',       'trainedNet'
        'RetinaAegis_CPU.mat',                'trainedNet'
        'RetinaAegis_Final_CPU.mat',          'trainedNet'
    };

    net = [];
    loadedFrom = '';
    for k = 1:size(modelFiles,1)
        if exist(modelFiles{k,1}, 'file')
            S = load(modelFiles{k,1}, modelFiles{k,2});
            if isfield(S, modelFiles{k,2})
                net = S.(modelFiles{k,2});
                loadedFrom = modelFiles{k,1};
                break;
            end
        end
    end

    if isempty(net)
        % Fallback: try any .mat with 'trainedNet'
        mats = dir('*.mat');
        for k = 1:numel(mats)
            try
                S = load(mats(k).name, 'trainedNet');
                if isfield(S, 'trainedNet')
                    net = S.trainedNet;
                    loadedFrom = mats(k).name;
                    break;
                end
            catch
                continue;
            end
        end
    end

    if isempty(net)
        error('Koi trained model nahi mila! Pehle Phase 2 training chalao.');
    end

    % Constants
    meanRGB = single([0.485 0.456 0.406]);
    stdRGB  = single([0.229 0.224 0.225]);
    
    grades = {
        'Grade 0 - No DR (Healthy)'
        'Grade 1 - Mild DR'
        'Grade 2 - Moderate DR'
        'Grade 3 - Severe DR'
        'Grade 4 - Proliferative DR'
    };
    
    gradeColors = [
        0.10 0.60 0.10   % Green - Healthy
        0.90 0.70 0.10   % Yellow - Mild
        0.95 0.50 0.10   % Orange - Moderate
        0.90 0.20 0.10   % Red-Orange - Severe
        0.75 0.05 0.05   % Dark Red - Proliferative
    ];

    %% ---- MAIN WINDOW ----
    fig = uifigure('Name', 'RETINA-AEGIS | Eye Checkup', ...
        'Position', [80 80 900 720], ...
        'Color', [0.97 0.97 0.98], ...
        'Resize', 'on');

    % Title Panel
    titlePanel = uipanel(fig, 'Position', [20 640 860 70], ...
        'BackgroundColor', [0.12 0.35 0.55], ...
        'BorderType', 'none');
    
    uilabel(titlePanel, 'Text', '👁️ RETINA-AEGIS', ...
        'Position', [20 10 400 45], 'FontSize', 28, ...
        'FontWeight', 'bold', 'FontColor', 'white');
    
    uilabel(titlePanel, 'Text', 'Diabetic Retinopathy Checkup', ...
        'Position', [440 15 400 35], 'FontSize', 16, ...
        'FontColor', [0.85 0.90 0.95], 'HorizontalAlignment', 'right');

    %% ---- LEFT PANEL: Upload & Instructions ----
    leftPanel = uipanel(fig, 'Position', [20 20 280 600], ...
        'BackgroundColor', 'white', 'BorderType', 'line', ...
        'BorderColor', [0.8 0.8 0.8], 'BorderWidth', 1);

    % Big Upload Button
    uploadBtn = uibutton(leftPanel, 'push', ...
        'Text', '📁 Upload Eye Photo', ...
        'Position', [20 520 240 60], ...
        'FontSize', 18, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.15 0.55 0.85], ...
        'FontColor', 'white', ...
        'ButtonPushedFcn', @(~,~) onUpload());

    % Status Box
    statusBox = uipanel(leftPanel, 'Position', [20 440 240 65], ...
        'BackgroundColor', [0.95 0.95 0.95], 'BorderType', 'line');
    
    statusIcon = uilabel(statusBox, 'Text', '⏳', ...
        'Position', [10 15 40 35], 'FontSize', 24);
    
    statusText = uilabel(statusBox, 'Text', 'No photo uploaded', ...
        'Position', [50 20 180 30], 'FontSize', 14, ...
        'FontColor', [0.4 0.4 0.4]);

    % Instructions (Simple Language)
    uilabel(leftPanel, 'Text', 'How to use:', ...
        'Position', [20 390 240 30], 'FontSize', 16, ...
        'FontWeight', 'bold', 'FontColor', [0.2 0.2 0.2]);

    steps = {
        '1. Click "Upload Eye Photo"'
        '2. Select a retina image'
        '3. Wait 2-3 seconds'
        '4. See result on right side'
    };
    for i = 1:4
        uilabel(leftPanel, 'Text', steps{i}, ...
            'Position', [30 360-(i-1)*35 230 28], ...
            'FontSize', 14, 'FontColor', [0.35 0.35 0.35]);
    end

    % Tips
    uilabel(leftPanel, 'Text', 'Tips:', ...
        'Position', [20 180 240 25], 'FontSize', 15, ...
        'FontWeight', 'bold', 'FontColor', [0.2 0.2 0.2]);
    
    tips = {
        '• Use clear, bright photo'
        '• Center the eye in image'
        '• Avoid blurry pictures'
    };
    for i = 1:3
        uilabel(leftPanel, 'Text', tips{i}, ...
            'Position', [30 155-(i-1)*28 230 25], ...
            'FontSize', 13, 'FontColor', [0.5 0.5 0.5]);
    end

    % Model Info
    uilabel(leftPanel, 'Text', sprintf('Model: %s', loadedFrom), ...
        'Position', [20 20 240 30], 'FontSize', 10, ...
        'FontColor', [0.6 0.6 0.6], 'WordWrap', 'on');

    %% ---- RIGHT PANEL: Results ----
    rightPanel = uipanel(fig, 'Position', [320 20 560 600], ...
        'BackgroundColor', 'white', 'BorderType', 'line', ...
        'BorderColor', [0.8 0.8 0.8], 'BorderWidth', 1);

    % Image Display Area
    imgPanel = uipanel(rightPanel, 'Position', [20 320 520 260], ...
        'BackgroundColor', [0.96 0.96 0.96], 'BorderType', 'line', ...
        'BorderColor', [0.85 0.85 0.85]);
    
    ax = uiaxes(imgPanel, 'Position', [10 10 500 240]);
    axis(ax, 'off');
    title(ax, 'Your Eye Photo will appear here', 'FontSize', 14, ...
        'Color', [0.6 0.6 0.6], 'FontWeight', 'normal');

    % Result Card
    resultCard = uipanel(rightPanel, 'Position', [20 140 520 160], ...
        'BackgroundColor', [0.98 0.98 0.98], 'BorderType', 'line', ...
        'BorderColor', [0.8 0.8 0.8], 'BorderWidth', 2);

    resultTitle = uilabel(resultCard, 'Text', 'Result:', ...
        'Position', [20 120 100 30], 'FontSize', 18, ...
        'FontWeight', 'bold', 'FontColor', [0.2 0.2 0.2]);

    % Big Grade Display
    gradeLabel = uilabel(resultCard, 'Text', 'Waiting for image...', ...
        'Position', [20 70 480 45], 'FontSize', 24, ...
        'FontWeight', 'bold', 'FontColor', [0.5 0.5 0.5]);

    % Confidence Bar Background
    confBg = uipanel(resultCard, 'Position', [20 35 480 20], ...
        'BackgroundColor', [0.9 0.9 0.9], 'BorderType', 'none');
    
    confBar = uipanel(confBg, 'Position', [0 0 0 20], ...
        'BackgroundColor', [0.15 0.55 0.85], 'BorderType', 'none');
    
    confText = uilabel(resultCard, 'Text', 'Confidence: --', ...
        'Position', [20 5 480 22], 'FontSize', 13, ...
        'FontColor', [0.4 0.4 0.4]);

    % Referable Alert Box
    alertBox = uipanel(rightPanel, 'Position', [20 20 520 100], ...
        'BackgroundColor', [0.95 0.95 0.95], 'BorderType', 'line', ...
        'BorderColor', [0.8 0.8 0.8], 'BorderWidth', 2);

    alertIcon = uilabel(alertBox, 'Text', '🏥', ...
        'Position', [20 25 60 50], 'FontSize', 36);
    
    alertTitle = uilabel(alertBox, 'Text', 'Doctor Advice:', ...
        'Position', [90 65 400 28], 'FontSize', 16, ...
        'FontWeight', 'bold', 'FontColor', [0.2 0.2 0.2]);
    
    alertMsg = uilabel(alertBox, 'Text', 'Upload a photo to see advice', ...
        'Position', [90 35 400 28], 'FontSize', 15, ...
        'FontColor', [0.4 0.4 0.4]);

    %% ---- CALLBACK: Upload & Predict ----
    function onUpload()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', ...
            'Eye Photos (*.jpg, *.png, *.tif)'}, ...
            'Select Eye Photo', 'MultiSelect', 'off');
        
        if isequal(file, 0)
            return; % User cancelled
        end

        % Update status
        statusIcon.Text = '⏳';
        statusText.Text = 'Checking...';
        statusText.FontColor = [0.15 0.55 0.85];
        drawnow;

        try
            fullPath = fullfile(path, file);
            
            % Read image
            img = imread(fullPath);
            if size(img,3) == 1
                img = repmat(img, [1 1 3]);
            elseif size(img,3) == 4
                img = img(:,:,1:3);
            end
            if ~isa(img, 'uint8')
                img = im2uint8(img);
            end

            % Show image
            cla(ax);
            imshow(img, 'Parent', ax);
            title(ax, file, 'Interpreter', 'none', 'FontSize', 12);
            axis(ax, 'off');

            % Preprocess
            imgResized = imresize(img, [224 224], 'bilinear');
            imgNorm = im2single(imgResized);
            imgNorm = (imgNorm - reshape(meanRGB,1,1,3)) ./ reshape(stdRGB,1,1,3);

            % Predict (CPU only — no gpu.dll error)
            scores = predict(net, imgNorm, 'ExecutionEnvironment', 'cpu');
            [confidence, predIdx] = max(scores);
            predGrade = predIdx - 1; % 0 to 4
            confPct = confidence * 100;

            % Update Grade Display
            gradeLabel.Text = grades{predIdx};
            gradeLabel.FontColor = gradeColors(predIdx, :);
            
            % Confidence Bar (visual)
            confText.Text = sprintf('AI Confidence: %.1f%%', confPct);
            confBar.Position = [0 0 min(confPct/100 * 480, 480) 20];
            
            % Color code confidence bar
            if confPct >= 80
                confBar.BackgroundColor = [0.10 0.60 0.10];
            elseif confPct >= 60
                confBar.BackgroundColor = [0.90 0.70 0.10];
            else
                confBar.BackgroundColor = [0.90 0.30 0.10];
            end

            % Referable Advice
            if predGrade >= 2
                % Severe/Proliferative — Red Alert
                alertBox.BackgroundColor = [1.00 0.92 0.92];
                alertBox.BorderColor = [0.80 0.20 0.20];
                alertIcon.Text = '⚠️';
                alertTitle.Text = 'Refer to Eye Doctor';
                alertTitle.FontColor = [0.75 0.10 0.10];
                alertMsg.Text = 'This needs specialist attention. Please visit an ophthalmologist soon.';
                alertMsg.FontColor = [0.60 0.10 0.10];
                statusIcon.Text = '⚠️';
                statusText.Text = 'Referable DR detected';
                statusText.FontColor = [0.75 0.10 0.10];
            else
                % No DR / Mild — Green Safe
                alertBox.BackgroundColor = [0.92 1.00 0.92];
                alertBox.BorderColor = [0.20 0.60 0.20];
                alertIcon.Text = '✅';
                alertTitle.Text = 'Routine Checkup OK';
                alertTitle.FontColor = [0.10 0.55 0.10];
                alertMsg.Text = 'Looks manageable. Continue regular checkups with your doctor.';
                alertMsg.FontColor = [0.10 0.50 0.10];
                statusIcon.Text = '✅';
                statusText.Text = 'Non-referable / Safe';
                statusText.FontColor = [0.10 0.55 0.10];
            end

        catch ME
            % Error handling — simple language
            statusIcon.Text = '❌';
            statusText.Text = 'Error! Try another photo.';
            statusText.FontColor = [0.75 0.10 0.10];
            
            gradeLabel.Text = 'Could not analyze';
            gradeLabel.FontColor = [0.75 0.10 0.10];
            
            alertBox.BackgroundColor = [1.00 0.95 0.90];
            alertBox.BorderColor = [0.80 0.50 0.20];
            alertIcon.Text = '⚠️';
            alertTitle.Text = 'Something went wrong';
            alertTitle.FontColor = [0.70 0.40 0.10];
            alertMsg.Text = sprintf('Error: %s. Please try a clearer photo.', ME.message);
            alertMsg.FontColor = [0.60 0.30 0.10];
            
            % Reset confidence
            confBar.Position = [0 0 0 20];
            confText.Text = 'Confidence: --';
        end
        
        drawnow;
    end
end