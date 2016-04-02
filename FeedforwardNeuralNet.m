%% Classification using Feedforward Neural Net

% One Hot Encoding
train_Y=dummyvar(train_Y);
test_Y=dummyvar(test_Y);

n=length(1:9);
m=length(0:.1:.9);
totaliterations=n*m;
counter=1;

for i=1:9 %search for best number of hidden units
    %units=[2^i 10]; % trying with more than one hidden layer
    units=2^i;
    net=patternnet(units);
    net.performFcn = 'crossentropy';
    %net.performFcn='mse'; % Try minimizing Mean square error 
    %net.layers{1}.transferFcn = 'logsig'; %Try logsig activation function
    net.performParam.regularization = 0.01; %Train with regularization
    net.trainFcn='traingdm'; %Train using steepest gradient descent w/ momentum
    net.trainParam.time	= 3;
    net.trainParam.epochs = 5000;
    net.trainParam.max_fail = 10000;
    net.divideparam.trainratio=0.9; %Train 90% of training data
    net.divideparam.valratio=0.1; %Validate with remaining 10% of training data
    for l=0:.1:.9 %search for best momentum value
        results(counter,3)=l;
        l=net.trainParam.mc;
        net = train(net,train_X',train_Y');
        O = net(train_X');
        perf = perform(net,train_Y',O);
        classes = vec2ind(O);
        [c] = confusion(train_Y',O);
        %[c,cm,ind,per] = confusion(train_Y',O);
        %figure, plotconfusion(train_Y',O)
        results(counter,1)=1-c;
        results(counter,2)=units(1);
        results(counter,4)=1-perf;
        counter=counter+1;
    end
end

[Acc I]=max(results(:,4)); %Calculate test classification accuracy
best_units=results(I,2); 
best_momentum=results(I,3);
fprintf('Highest CV Accuracy=%g, with hidden units=%g and momentum=%g', Acc, best_units, best_momentum);

%% Testing using our best parameters
%net=patternnet([best_units 10]);
net=patternnet(best_units);
%net.performFcn='mse'; %Mean square error
net.performFcn = 'crossentropy';

net.trainFcn='traingdm'; %Train best model using steepest gradient descent w/ momentum
%net.layers{1}.transferFcn = 'logsig'; %Train best model using logsig activation function
net.trainParam.time	= 3;
net.trainParam.epochs = 5000;
net.trainParam.max_fail = 10000;
net.divideparam.trainratio=1; %Not performing cross validation, using all test data
net.trainParam.mc=best_momentum;
net = train(net,test_X',test_Y');
O_test = net(test_X'); %Use best model on test data
perf_test = perform(net,test_Y',O_test);
test_classes = vec2ind(O_test);
[c_test] = confusion(test_Y',O_test);
figure, plotconfusion(train_Y',O)
test_accuracy_c=(1-c_test)*100; %Calculate test classification accuracy