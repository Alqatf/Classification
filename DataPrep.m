%Normalize Data
X=(X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

%Shuffle and split into 80% Train, 20% Test
rand_ind=randperm(length(X))';
n=length(X);
perc=.8; %Percent of samples used for training
perc_train=round(perc*n); %Number of samples used in training
perc_test=perc_train+1;

train_ind=rand_ind(1:perc_train,1); 
test_ind=rand_ind(perc_test:end,1); 
train_X=X((train_ind(:,1)),:);
train_Y=Y((train_ind(:,1)),:);
test_X=X((test_ind(:,1)),:);
test_Y=Y((test_ind(:,1)),:);
