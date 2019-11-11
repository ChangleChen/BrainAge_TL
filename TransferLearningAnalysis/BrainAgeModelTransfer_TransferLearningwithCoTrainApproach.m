%% dMRI-based BrainAge Model: Transfer Learning with Co-Train Approach
% Developed by C.L. Chen
%% Notes
% The CamCAN cohort is defined as the source domain
% The NTUH, HH, and Guys cohorts are defined as the target domains
%% Load the tensor-based features from target domains
% NTUH cohort
load('Data_proc_tensorfeature_NTUH_Train_Test_N300_N105.mat')
% HH cohort
% load('Data_proc_tensorfeature_HH_Train_Test_N120_N56.mat')
% Guys cohort
% load('Data_proc_tensorfeature_Guys_Train_Test_N120_N64.mat')
fprintf('The data are allocated!\n');
%% Load the advanced features from target domains
% NTUH cohort
load('Data_proc_advfeature_NTUH_Train_Test_N300_N105.mat')
% HH cohort
% load('Data_proc_advfeature_HH_Train_Test_N120_N56.mat')
% Guys cohort
% load('Data_proc_advfeature_Guys_Train_Test_N120_N64.mat')
fprintf('The data are allocated!\n');
%% Specify the features and responses for the training and test sets from the target domain
x_tune_target = x_ntuh_af_training; % the input feature for training set
y_tune_target = y_ntuh_training; % the response for training set
x_test_target = x_ntuh_af_test; % the input feature for test set
y_test_target = y_ntuh_test; % the response for test set
x_tune_target(:,end) = 2; % change the site indicator (default: CamCAN: 1)
x_test_target(:,end) = 2; % change the site indicator (default: CamCAN: 1)
%% Using the random subset to fine-tune the model
[x_tune_target,y_tune_target] = random_select(x_tune_target,y_tune_target,200);
% the first to third input argument is 
% the predictors, response, and selected tuning sample size, respectively
fprintf('Random Selection: Done!\n')
%% Load the features from source domain
% CamCAN "tensor" features
% load('Data_proc_tensorfeature_CamCAN_Train_Test_N500_N116.mat')
% CamCAN "advanced" features
load('Data_proc_advfeature_CamCAN_Train_Test_N500_N116.mat')
x_tune_source = x_cc_af_training; % CamCAN site indicator: 1
y_tune_source = y_cc_training;
x_test_source = x_cc_af_test; % CamCAN site indicator: 1
y_test_source = y_cc_test;
%% Integrate those two datasets from both source and target domains
x_tune = [x_tune_source;x_tune_target];
y_tune = [y_tune_source;y_tune_target];
%% Load the pretrained model from source domain
% make sure the pretrained model with the same feature types
% CamCAN pretrained model using "tensor" features
% load('Pretrained_Model_CamCAN_usingTensorFeatures.mat')
% CamCAN pretrained model using "advanced" features
load('Pretrained_Model_CamCAN_usingAdvancedFeatures.mat')

%% Transfer Learning Approach
pt_model = Model_cc; % specify the pre-trained model

%%%   pre-trained model settings  %%%
% Optimizer
pt_model.trainFcn = 'traingdx'; % 'trainscg' is faster
% Epochs
pt_model.trainParam.epochs = 700; 
% Tolerance of early stopping
pt_model.trainParam.max_fail = 500; % to force the model going through at least 500 epos
% Verbose
pt_model.trainParam.showWindow = 1;
% Loss function
pt_model.performFcn = 'mse'; % mae, mse, sae, sse, etc.
% Regularization
pt_model.performParam.regularization = 0.01; % 0.25, 0.1, 0.01, ect.
% Freeze partial layers or not (demo)
fot = logical([0,0,0,0,0,0]); % specify which layers should be frozen (frozen: 1).
% make sure the length of the above vector have the same size with the
% number of layers in the model (use Model.numLayers to check)
% [0,0,0,1,1,1] means the fourth to the last layers (close to the input layer) are frozen.
pt_model = freezelayer(pt_model,fot);

Model_TL = train(pt_model,x_tune',y_tune','useGPU','yes'); % training

%% Model inference
% Prediction on two sites at once

% Target domain
y_hat_tune = Model_TL(x_tune_target')';
model_inference(y_hat_tune,y_tune_target,1);
y_hat_test = Model_TL(x_test_target')';
model_inference(y_hat_test,y_test_target,2);

% Source domain
y_hat_tune = Model_TL(x_tune_source')';
model_inference(y_hat_tune,y_tune_source,3);
y_hat_test = Model_TL(x_test_source')';
model_inference(y_hat_test,y_test_source,4);
%% Function
function [x_tune,y_tune] = random_select(x_tune,y_tune,samplesize)
figure(999); set(gcf,'color','w','Position',[50,50,900,600]);
fig = gcf;
fig.PaperPositionMode = 'auto';
histogram(y_tune,15); hold on;
dice1 = randperm(length(y_tune));
x_tune = x_tune(dice1(1:samplesize),:);
y_tune = y_tune(dice1(1:samplesize));
histogram(y_tune,15); legend('original','selected');
xlabel('Age (years)'); ylabel('Counts');
end

function pt_model = freezelayer(pt_model,fot)
% freeze layers in the model
% check the consistency b/w pt_model and fot
if pt_model.numLayers == length(fot)
    if fot(1) == true
        for ly = 1:numel(pt_model.inputWeights)
            pt_model.inputWeights{ly}.learn = false;
        end
        pt_model.biases{1}.learn = false;
    else
        for ly = 1:numel(pt_model.inputWeights)
            pt_model.inputWeights{ly}.learn = true;
        end
        pt_model.biases{1}.learn = true;
    end
    
    for arg = 2:length(fot)
        if fot(arg) == true
            for ly = 1:numel(pt_model.layerWeights(arg,:))
                try
                    pt_model.layerWeights{arg,ly}.learn = false;
                end
            end
            pt_model.biases{arg}.learn = false;
        else
            for ly = 1:numel(pt_model.layerWeights(arg,:))
                try
                    pt_model.layerWeights{arg,ly}.learn = true;
                end
            end
            pt_model.biases{arg}.learn = true;
        end
    end
else
    error('The dimension of the freezable vector is not consistent with the number of layers in the model!');
end
end


function model_inference(predict_age,true_age,k)
cuscolormap = [linspace(0,0.2,128)',linspace(0,1,128)',linspace(1,0.2,128)';...
    linspace(0.2,0,128)',linspace(1,0,128)',linspace(0.2,1,128)'];
fprintf('Rho: %g, RMSE: %g, MAE: %g \n',...
    corr(predict_age,true_age),sqrt(mean((true_age-predict_age).^2)),mean(abs(true_age-predict_age)));

if nargin == 3
    figure(k); set(gcf,'color','w','Position',[200,200,900,600]);
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    plot(linspace(10.5,95.5,200),linspace(10.5,95.5,200),'color',[0.8,0.8,0.8],...
        'linewidth',6,'LineStyle','-.'); hold on;
    tPAD = predict_age-true_age;
    scatter(true_age,predict_age,155,tPAD,'filled','MarkerEdgeColor',[0,0,0]);
    colorbar; colormap(cuscolormap)
    xlim([12,95]); ylim([12,95])
    xlabel('Chronological Age (years)'); ylabel('Predicted Age (years)');
    title(sprintf('PAD score: %g',mean(tPAD)));
    caxis([-25,25]);
    set(gca,'fontsize',14,'fontweight','bold'); grid on;
    set(gca,'color',[0.95,0.95,0.95]);
    [r,p]=corr(true_age,tPAD);
    fprintf('Age-related bias: %g, with p-value %g \n',r,p)
    fprintf('Mean PAD: %g ... \n',mean(tPAD));
end
end

