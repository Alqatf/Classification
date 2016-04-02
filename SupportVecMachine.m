%% Classification with Support Vector Machines
%Using LIBSVM and plotroc https://www.csie.ntu.edu.tw/~cjlin/libsvm/, https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/roc/plotroc.m 

log2c_list = -1:4; %Initialize cost hyperparameter
log2g_list = -3:3; %Initialize gamma hyperparameter

% Use Linear Kernel

numLog2c = length(log2c_list);
cvMatrix = zeros(numLog2c,1);
bestAccuracy = 0;
totalRuns = numLog2c;

runCounter = 1;
for i = 1:numLog2c %Loop through cost hyperparameter
        log2c = log2c_list(i);
        param = ['-q -t 0 -v 10 -c ', num2str(2^log2c)]; %Initialize model parameters, 10-fold Cross Val
        accuracy=svmtrain(train_Y,train_X,param); %Train model
        LincvMatrix(i,1) = accuracy; %Store accuracy
        LincvMatrix(i,2) = 2^log2c; %Store cost value
        if (accuracy >= bestAccuracy), 
            bestAccuracy = accuracy; 
            bestLog2c = 2^log2c; 
        end
        runCounter = runCounter+1;
        fprintf('(best: c=%g,cv=%g)\n', bestLog2c, bestAccuracy);
end

disp(['best log2c:',num2str(bestLog2c),' accuracy:',num2str(bestAccuracy),'%']);

% Test
param = ['-t 0 -c ', num2str(bestLog2c)]; %Test using best cost value found during training
best_model1 = svmtrain(train_Y, train_X, param); %Train the best model
numSV=best_model1.nSV; %# of Support Vector Machines per class
[Lpredict_label, Lin_test_accuracy, prob_values] = svmpredict(test_Y, test_X, best_model1); % Test the best model
Lin_test_accuracy, bestLog2c, numSV
Lin_Stats=confusionmatStats(test_Y,Lpredict_label)
hold on;
subplot(2,2,1)
plotroc(test_Y,Lpredict_label,best_model1) %Plot ROC Curve

% Use polynomial Kernel (degree 2)
bestcv = 0;

numLog2c = length(log2c_list); 
numLog2g = length(log2g_list);
iteration = 1;
totaliterations=numLog2c*numLog2g;

for i = 1:numLog2c %Loop through cost hyperparameter
  log2c = log2c_list(i); 
  for i =1:numLog2g %Loop through gamma hyperparameter
    log2g = log2g_list(i);
    parameters = ['-q -t 1 -d 2 -v 10 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(train_Y, train_X, parameters);
    if (cv >= bestcv),
      bestcv = cv; 
      bestc = 2^log2c; 
      bestg = 2^log2g;
    end
    fprintf('(best: c=%g, g=%g, cv=%g)\n', bestc, bestg, bestcv);
    iteration = iteration+1;
  end
end

parameters = ['-q -t 1 -d 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];
best_model2 = svmtrain(train_Y, train_X, parameters);
numSV=best_model2.nSV;
[predict_label2, PolD2_test_accuracy, decision_values] = svmpredict(test_Y, test_X, best_model2);
PolD2_test_accuracy,bestc, bestg, numSV
D2_Stats=confusionmatStats(test_Y,predict_label2)
subplot(2,2,2)
plotroc(test_Y,predict_label2,best_model2)

% Use polynomial Kernel (degree 3)
bestcv = 0;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
iteration = 1;
totaliterations=numLog2c*numLog2g;

for i = 1:numLog2c
  log2c = log2c_list(i);
  for i =1:numLog2g
    log2g = log2g_list(i);
    parameters = ['-q -t 1 -v 10 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(train_Y, train_X, parameters);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('(best: c=%g, g=%g, cv=%g)\n', bestc, bestg, bestcv);
    iteration = iteration+1;
  end
end

parameters = ['-q -t 1 -c ', num2str(bestc), ' -g ', num2str(bestg)];
best_model3 = svmtrain(train_Y, train_X, parameters);
numSV=best_model3.nSV;
[predicted_label3, PolD3_test_accuracy, decision_values] = svmpredict(test_Y, test_X, best_model3);
PolD3_test_accuracy,bestc, bestg, numSV
D3_Stats=confusionmatStats(test_Y,predicted_label3)
subplot(2,2,3)
plotroc(test_Y,predicted_label3,best_model3)

% Use RBF Kernel

bestcv = 0;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
iteration = 1;
totaliterations=numLog2c*numLog2g;

for i = 1:numLog2c
  log2c = log2c_list(i);
  for i =1:numLog2g
    log2g = log2g_list(i);
    parameters = ['-q -t 2 -v 10 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(train_Y, train_X, parameters);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('(best: c=%g, g=%g, cv=%g)\n', bestc, bestg, bestcv);
    iteration = iteration+1;
  end
end

parameters = ['-q -t 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];
best_model4 = svmtrain(train_Y, train_X, parameters);
numSV=best_model4.nSV;
[predicted_label4, RBF_test_accuracy, decision_values] = svmpredict(test_Y, test_X, best_model4);
RBF_test_accuracy,bestc, bestg, numSV

RBFstats=confusionmatStats(test_Y,predicted_label4)
subplot(2,2,4)
plotroc(test_Y,predicted_label4,best_model4)
hold off
