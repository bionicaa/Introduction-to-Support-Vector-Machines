% SVM Example for 'An Introduction to Support Vector Machines' 
% Author: Amin Abdi
% Date: 17/08/2020
%
% Binary classification of cats and dogs using parameters of weight and
% length.

%%

%import training dataset
tr_data = csvread('training_data(svm).csv');
%tr_data = svmdata;

%prepare data
x1 = tr_data(:,2); %column 2 - length
y1 = tr_data(:,1); %column 1 - weight 
sz = 30; %size of data plots

%plot the training data for visualization
figure
scatter(x1,y1,sz,'filled')
title('Visualization of Training data')
xlabel('Length (in)')
ylabel('Weight (kg)')


%%

%fit model

X = [x1,y1]; %predictors
Y = tr_data(:,3);

SVM = fitcsvm(X,Y, 'ClassNames', {'cats', 'dogs'});

%design hyperplane
w1 = (SVM.Alpha' * SVM.SupportVectors{:,1});
w2 = (SVM.Alpha' * SVM.SupportVectors{:,2});
bias = SVM.Bias;
a = -w1/w2; %gradient
b = -SVM.Bias/w2; %margin

h = a*X + b; %y=mx+c

%visualize SVM with decision boundary
figure
scatter(tr_data{:,2},tr_data{:,1},'bo','filled' );
hold on
plot(h,'g','MarkerSize',10);
hold on
xlim([0 8])
title('SVM classifier')
xlabel('Length (in)')
ylabel('Weight (kg)')
hold off

%%

%prediction of new data, only run after SVM is run. 

%read force sensor output
l = input('What is the length of the animal?');
%input weight of object
w = input('What is the weight of the animal?');


%predict new case
n_data = [w, l]; 
label = predict(SVM,n_data);
disp(label)
%%

%visualise to determine correct classification
figure 
scatter(tr_data(:,2),tr_data(:,1),'bo','filled');
hold on
plot(h,'g','MarkerSize',10);
hold on
scatter(n_data(:,2),n_data(:,1),'r','filled');
hold on
title('SVM classifier')
xlabel('Length (in)')
ylabel('Weight (kg)')
hold off



%%

clear all