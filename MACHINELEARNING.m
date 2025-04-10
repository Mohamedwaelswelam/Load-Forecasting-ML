%% SVM for Power and Gas Load Prediction with PolynomialOrder=2
%  Systematic search over BoxConstraint in [0.1, 1, 10, 100], Epsilon in [0.001..].
%  Then pick the best BC, Eps, Kernel via cross-validation, re-train fully, and predict 2026 & 2027.
%  If the top CV MSE yields 2026 < 2025, try next best, etc.

clc; clear; close all;
rng(42); % Fix randomness

%% 1) 15-Year Data Range
years = (2011:2025)';
future_years = (2026:2027)';

%% 2) Power Load Data (MW), 15x15 (unchanged)
power_load = [
    0.0000 16.2031 21.2594 57.4187 18.0527 45.0754 66.4336 25.8973 30.7303 20.0813 45.0138 21.3960 15.5300 31.1737 52.5188
    0.0000 18.3635 24.0940 65.0745 20.4597 51.0874 75.2904 29.3503 34.8276 22.7593 51.4156 24.2488 17.6017 35.3302 59.5213
    0.0000 20.5240 26.9315 72.7303 22.8667 57.0954 84.1512 32.8033 38.9250 25.8414 57.0175 27.1016 19.6734 39.4871 66.5280
    0.0000 22.6844 29.7690 80.3861 25.2737 63.1034 93.0120 36.2563 43.0231 28.9235 62.6194 29.9544 21.7451 43.6440 73.5343
    0.0000 24.0448 32.6065 88.0419 27.6807 69.1114 102.8730 39.7093 47.1212 32.0056 68.2213 32.8072 23.8168 47.8009 80.5400
    0.0000 27.0052 35.4323 95.6978 30.0878 75.1256 110.7226 43.1622 51.2171 35.0877 75.0230 35.6600 25.8834 51.9562 87.5313
    0.0000 29.0609 39.6878 100.3807 31.5426 82.4204 114.1331 46.0215 53.2105 37.1698 82.5701 39.6159 27.7163 54.1356 93.0367
    0.0000 31.0341 43.8364 105.2176 32.9974 89.0422 117.6498 49.2148 55.1219 39.2519 90.1860 43.5718 29.8514 56.0815 98.4230
    0.0000 32.9570 47.5025 110.1294 34.6618 96.9418 121.1377 52.4081 57.3768 41.3340 97.5131 47.5277 31.9387 57.9435 104.0632
    0.0000 34.9805 51.1336 114.8914 35.9070 101.3048 124.3646 55.6014 58.9447 43.4161 105.2111 51.4836 33.2150 59.9733 109.7034
    0.0000 36.9537 55.2822 119.7283 37.3618 107.5996 127.7751 58.7947 60.8561 45.4982 112.7581 55.4395 35.0479 61.9192 115.3436
    0.0000 38.9269 59.4308 124.5652 38.8166 113.8944 131.1856 61.9880 62.7675 47.5803 120.3051 59.3954 36.8808 63.8650 120.9838
    0.0000 40.9001 63.5794 129.4021 40.2714 120.1892 134.5961 65.1813 64.6789 49.6624 127.8521 63.3513 38.7137 65.8108 126.6240
    0.0000 42.8733 67.7280 134.2390 41.7262 126.4840 138.0066 68.3746 66.5903 51.7445 135.3991 67.3072 40.5466 67.7566 132.2642
    0.0000 44.1000 70.0000 140.0000 44.1000 140.0000 140.0000 70.0000 70.0000 53.8266 140.0000 70.0000 44.1000 70.0000 140.0000
];

%% 3) Gas Load Data (MMCFD), 15x14 (unchanged)
gas_load = [
    0.0000  11.0225  27.3335  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  45.1147  16.3743  28.9770
    0.0000  12.4943  30.9771  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  51.1225  18.5615  32.6253
    0.0000  13.9661  34.6207  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  57.1303  20.7487  36.2737
    0.0000  15.4379  38.2643  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  63.1381  22.9359  39.9221
    0.0000  16.9097  41.9079  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  69.1459  25.1231  43.5705
    0.0000  18.3709  45.5558  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  75.1911  27.2905  48.2950
    0.0000  19.7693  51.0272  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  78.8705  28.6101  52.9845
    0.0000  21.1116  56.3611  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  82.6710  29.8998  57.2414
    0.0000  22.4198  61.0747  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  86.5303  31.4393  62.3197
    0.0000  23.6143  65.7578  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  90.4607  32.9835  66.7873
    0.0000  24.8089  70.4425  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  94.6522  34.4832  71.6339
    0.0000  25.9783  75.2480  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  98.3793  35.8406  76.1511
    0.0000  27.3551  80.2742  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 102.3558  37.1396  80.8640
    0.0000  28.6638  85.2075  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 106.3944  38.6043  85.6018
    0.0000  30.0000  90.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 110.0000  40.0000  90.0000
];

num_buses = size(power_load,2);
num_nodes = size(gas_load,2);

% Grid
boxVals = [0.1, 1, 10, 100];
kernels = {'linear','polynomial','gaussian'};
epsilonVals = [0.001, 0.01, 0.1, 1];
kFolds = 5;

%% Arrays to store final 2026 predictions (for plotting)
all_predicted_2026_power = zeros(1, num_buses);
all_predicted_2026_gas   = zeros(1, num_nodes);

%% 4) Train SVM for Power Load
disp('=== MSE Calculations for Power Load ===');
for i = 2:num_buses
    yTrain = power_load(:, i);

    bestKernel = '';
    bestBC = NaN;
    bestEps = NaN;
    bestMSE = Inf;

    % We'll store all combos in allResults to pick from if we want growth
    allResults = []; % struct array with fields: kernel, bc, eps, mse

    fprintf('\nBus %d MSE values:\n', i);

    % 4a) Search
    for kernel = kernels
        for bC = boxVals
            for epsVal = epsilonVals

                if strcmp(kernel{1}, 'polynomial')
                    svmTemp = fitrsvm(years, yTrain, ...
                        'KernelFunction', kernel{1}, ...
                        'PolynomialOrder', 2, ...
                        'Standardize', true, ...
                        'BoxConstraint', bC, ...
                        'Epsilon', epsVal, ...
                        'KFold', kFolds);
                else
                    svmTemp = fitrsvm(years, yTrain, ...
                        'KernelFunction', kernel{1}, ...
                        'Standardize', true, ...
                        'BoxConstraint', bC, ...
                        'Epsilon', epsVal, ...
                        'KFold', kFolds);
                end

                thisMSE = kfoldLoss(svmTemp);
                fprintf('Kernel=%s | BoxC=%.4g | Eps=%.4g | CV MSE=%.4f\n', ...
                    kernel{1}, bC, epsVal, thisMSE);

                % track best
                if thisMSE < bestMSE
                    bestMSE = thisMSE;
                    bestBC = bC;
                    bestKernel = kernel{1};
                    bestEps = epsVal;
                end

                % store in allResults
                s.kernel = kernel{1};
                s.bc = bC;
                s.eps = epsVal;
                s.mse = thisMSE;
                allResults = [allResults; s]; %#ok<AGROW>
            end
        end
    end

    % Done searching
    fprintf('Lowest CV MSE: Kernel=%s | BoxC=%.4g | Eps=%.4g | MSE=%.4f (SELECTED)\n', ...
        bestKernel, bestBC, bestEps, bestMSE);

    % 4b) Train final model with the top MSE to keep original code
    if strcmp(bestKernel, 'polynomial')
        SVMModel_Power = fitrsvm(years, yTrain, ...
            'KernelFunction', bestKernel, ...
            'PolynomialOrder', 2, ...
            'Standardize', true, ...
            'BoxConstraint', bestBC, ...
            'Epsilon', bestEps);
    else
        SVMModel_Power = fitrsvm(years, yTrain, ...
            'KernelFunction', bestKernel, ...
            'Standardize', true, ...
            'BoxConstraint', bestBC, ...
            'Epsilon', bestEps);
    end

    pred_power = predict(SVMModel_Power, future_years);
    final_predictions = predict(SVMModel_Power, years);
    final_MSE = mean((final_predictions - yTrain).^2);

    fprintf('Bus %d | best Kernel=%s | best BoxC=%.4g | best Eps=%.4g | 2025=%.2f MW | 2026=%.2f MW | 2027=%.2f MW | CV MSE=%.4f | Final MSE=%.4f\n',...
        i, bestKernel, bestBC, bestEps, yTrain(end), pred_power(1), pred_power(2), bestMSE, final_MSE);

    % 4c) Extra Step: see if top method yields 2026 >= 2025
    if pred_power(1) >= yTrain(end)
        % Good, we already have growth
        fprintf('   Already grows. Using this top CV MSE method.\n');
    else
        % We'll look from best to worse MSE, see if we can find a method
        % that yields predicted(2026) >= predicted(2025).
        fprintf('   The top CV MSE method yields 2026 < 2025. Trying next best...\n');

        % sort all results ascending by mse
        [~, idxSort] = sort([allResults.mse]);
        chosenRank = NaN; 
        foundGrowth = false;

        for rankCandidate = 1:numel(idxSort)
            rr = allResults(idxSort(rankCandidate));
            % Train a final model for that combo
            if strcmp(rr.kernel, 'polynomial')
                tmpModel = fitrsvm(years, yTrain, ...
                    'KernelFunction', rr.kernel, ...
                    'PolynomialOrder', 2, ...
                    'Standardize', true, ...
                    'BoxConstraint', rr.bc, ...
                    'Epsilon', rr.eps);
            else
                tmpModel = fitrsvm(years, yTrain, ...
                    'KernelFunction', rr.kernel, ...
                    'Standardize', true, ...
                    'BoxConstraint', rr.bc, ...
                    'Epsilon', rr.eps);
            end

            preds = predict(tmpModel, future_years);
            y2025 = yTrain(end);
            y2026 = preds(1);

            if y2026 >= y2025
                % Found a method that grows
                foundGrowth = true;
                chosenRank = rankCandidate; %#ok<NASGU>
                fprintf('   *** We choose Kernel=%s|C=%.4g|Eps=%.4g ***\n', rr.kernel, rr.bc, rr.eps);
                fprintf('   Because it causes growth (2026=%.2f >= 2025=%.2f) \n', y2026, y2025);
                fprintf('   and is the #%d best in ascending MSE order.\n', rankCandidate);
                break;
            end
        end

        if ~foundGrowth
            fprintf('   No method in ascending MSE forced 2026 >= 2025.\n');
            fprintf('   We keep the top CV MSE method anyway.\n');
        end
    end

    % Store the final top method's 2026 prediction for plotting:
    all_predicted_2026_power(i) = pred_power(1);
end

%% 5) Train SVM for Gas Load (same approach)
disp('=== MSE Calculations for Gas Load ===');
for j = 2:num_nodes
    yTrain = gas_load(:, j);

    bestKernel = '';
    bestBC = NaN;
    bestEps = NaN;
    bestMSE = Inf;

    allResults = [];

    fprintf('\nNode %d MSE values:\n', j);

    % Search
    for kernel = kernels
        for bC = boxVals
            for epsVal = epsilonVals

                if strcmp(kernel{1}, 'polynomial')
                    svmTemp = fitrsvm(years, yTrain, ...
                        'KernelFunction', kernel{1}, ...
                        'PolynomialOrder', 2, ...
                        'Standardize', true, ...
                        'BoxConstraint', bC, ...
                        'Epsilon', epsVal, ...
                        'KFold', kFolds);
                else
                    svmTemp = fitrsvm(years, yTrain, ...
                        'KernelFunction', kernel{1}, ...
                        'Standardize', true, ...
                        'BoxConstraint', bC, ...
                        'Epsilon', epsVal, ...
                        'KFold', kFolds);
                end

                thisMSE = kfoldLoss(svmTemp);
                fprintf('Kernel=%s | BoxC=%.4g | Eps=%.4g | CV MSE=%.4f\n', ...
                    kernel{1}, bC, epsVal, thisMSE);

                if thisMSE < bestMSE
                    bestMSE = thisMSE;
                    bestBC  = bC;
                    bestKernel = kernel{1};
                    bestEps = epsVal;
                end

                s.kernel = kernel{1};
                s.bc = bC;
                s.eps = epsVal;
                s.mse = thisMSE;
                allResults = [allResults; s]; %#ok<AGROW>
            end
        end
    end

    fprintf('Lowest CV MSE: Kernel=%s | BoxC=%.4g | Eps=%.4g | MSE=%.4f (SELECTED)\n', ...
        bestKernel, bestBC, bestEps, bestMSE);

    % Train final model
    if strcmp(bestKernel, 'polynomial')
        SVMModel_Gas = fitrsvm(years, yTrain, ...
            'KernelFunction', bestKernel, ...
            'PolynomialOrder', 2, ...
            'Standardize', true, ...
            'BoxConstraint', bestBC, ...
            'Epsilon', bestEps);
    else
        SVMModel_Gas = fitrsvm(years, yTrain, ...
            'KernelFunction', bestKernel, ...
            'Standardize', true, ...
            'BoxConstraint', bestBC, ...
            'Epsilon', bestEps);
    end

    pred_gas = predict(SVMModel_Gas, future_years);
    final_predictions = predict(SVMModel_Gas, years);
    final_MSE = mean((final_predictions - yTrain).^2);

    fprintf('Node %d | best Kernel=%s | best BoxC=%.4g | best Eps=%.4g | 2025=%.2f | 2026=%.2f | 2027=%.2f | CV MSE=%.4f | Final MSE=%.4f\n',...
        j, bestKernel, bestBC, bestEps, yTrain(end), pred_gas(1), pred_gas(2), bestMSE, final_MSE);

    % Growth check
    if pred_gas(1) >= yTrain(end)
        fprintf('   Already grows. Using this top CV MSE method.\n');
    else
        fprintf('   The top CV MSE method yields 2026 < 2025. Trying next best...\n');

        [~, idxSort] = sort([allResults.mse]);
        foundGrowth = false;

        for rankCandidate = 1:numel(idxSort)
            rr = allResults(idxSort(rankCandidate));
            % train final
            if strcmp(rr.kernel, 'polynomial')
                tmpModel = fitrsvm(years, yTrain, ...
                    'KernelFunction', rr.kernel, ...
                    'PolynomialOrder', 2, ...
                    'Standardize', true, ...
                    'BoxConstraint', rr.bc, ...
                    'Epsilon', rr.eps);
            else
                tmpModel = fitrsvm(years, yTrain, ...
                    'KernelFunction', rr.kernel, ...
                    'Standardize', true, ...
                    'BoxConstraint', rr.bc, ...
                    'Epsilon', rr.eps);
            end

            preds = predict(tmpModel, future_years);
            y2025 = yTrain(end);
            y2026 = preds(1);

            if y2026 >= y2025
                foundGrowth = true;
                fprintf('   *** We choose Kernel=%s|C=%.4g|Eps=%.4g ***\n', rr.kernel, rr.bc, rr.eps);
                fprintf('   Because it causes growth (2026=%.2f >= 2025=%.2f)\n', y2026, y2025);
                fprintf('   and is the #%d best in ascending MSE order.\n', rankCandidate);
                fprintf('   2027 Prediction: %.2f\n', preds(2));
                final_predictions = predict(tmpModel, years);
                final_MSE = mean((final_predictions - yTrain).^2);
                fprintf('   CV MSE: %.4f | Final MSE: %.4f\n', rr.mse, final_MSE);
                break;
            end
        end

        if ~foundGrowth
            fprintf('   No method forced 2026 >= 2025. Keeping top CV MSE method.\n');
        end
    end

    % Store the final top method's 2026 prediction for plotting:
    all_predicted_2026_gas(j) = pred_gas(1);
end

%% ========== EXTRA: Plot 2 Graphs (Buses & Nodes) ==========
% -- Plot for Buses (Power Loads) --
figure;
hold on;
legendEntries = {};
for i = 2:num_buses
    if all_predicted_2026_power(i) ~= 0
        plot([years; 2026], [power_load(:, i); all_predicted_2026_power(i)], '-o');
        legendEntries{end+1} = ['Bus ' num2str(i)]; %#ok<AGROW>
    end
end
title('Power Buses - Historical Load (2011-2025) + Predicted 2026');
xlabel('Year');
ylabel('Power Load (MW)');
if ~isempty(legendEntries)
    legend(legendEntries, 'Location', 'best');
end
hold off;

% -- Plot for Nodes (Gas Loads) --
figure;
hold on;
legendEntries = {};
for j = 2:num_nodes
    if all_predicted_2026_gas(j) ~= 0
        plot([years; 2026], [gas_load(:, j); all_predicted_2026_gas(j)], '-o');
        legendEntries{end+1} = ['Node ' num2str(j)]; %#ok<AGROW>
    end
end
title('Gas Nodes - Historical Load (2011-2025) + Predicted 2026');
xlabel('Year');
ylabel('Gas Load (MMCFD)');
if ~isempty(legendEntries)
    legend(legendEntries, 'Location', 'best');
end
hold off;
