%% K Nearest Neighbors Classifier Implementation

neighbor=20; % # of Neighbors to loop through

% Using euclidean distance metric

for i=1:neighbor % Grid search for best number of neighbors
knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',i, 'DistanceWeight','inverse');
CV_knn_mdl = crossval(knn_mdl,'KFold',10); %10-Fold CV
kloss(i) = kfoldLoss(CV_knn_mdl);
B=sprintf('Iteration %d/20',i);
display(B)
end
[M,I] = min(kloss(:)); %Find which neighbor gives the smallest kfoldLoss  
D=sprintf('Best euclidean performance occurs with %d neighbors.',I);
display(D)

%Test using best parameters 
best_knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',I,'DistanceWeight','inverse'); %Create new model with best neighbor
labels= predict(best_knn_mdl,test_X); %Predict the test labels
accuracy=find(labels==test_Y); 
TestAccuracy=length(accuracy)/length(test_Y) %Calculate Test Classification Accuracy
Error=length(labels)-length(accuracy)

% Using chebychev distance metric

for i=1:neighbor %loop to find best number of neighbors
knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',i,'Distance','chebychev', 'DistanceWeight','inverse');
CV_knn_mdl = crossval(knn_mdl,'KFold',10); %10-Fold CV
kloss(i) = kfoldLoss(CV_knn_mdl);
B=sprintf('Iteration %d/20',i);
display(B)
end
[M,I] = min(kloss(:)); %Find which neighbor gives the smallest kfoldLoss  
D=sprintf('Best chebychev performance occurs with %d neighbors.',I);
display(D)

best_knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',I,'Distance','chebychev', 'DistanceWeight','inverse'); %Create new model with best neighbor
labels= predict(best_knn_mdl,test_X);
accuracy=find(labels==test_Y); 
TestAccuracy=length(accuracy)/length(test_Y)
Error=length(labels)-length(accuracy)

% Using mahalanobis distance metric

for i=1:neighbor %loop to find best number of neighbors
knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',i,'Distance','mahalanobis', 'DistanceWeight','inverse');
CV_knn_mdl = crossval(knn_mdl,'KFold',10); %10-Fold CV
kloss(i) = kfoldLoss(CV_knn_mdl);
B=sprintf('Iteration %d/20',i);
display(B)
end
[M,I] = min(kloss(:)); %Find which neighbor gives the smallest kfoldLoss  
D=sprintf('Best mahalanobis performance occurs with %d neighbors.',I);
display(D)

best_knn_mdl = fitcknn(train_X,train_Y,'NumNeighbors',I,'Distance','mahalanobis', 'DistanceWeight','inverse'); %Create new model with best neighbor
labels= predict(best_knn_mdl,test_X);
accuracy=find(labels==test_Y); 
TestAccuracy=length(accuracy)/length(test_Y)
Error=length(labels)-length(accuracy)




