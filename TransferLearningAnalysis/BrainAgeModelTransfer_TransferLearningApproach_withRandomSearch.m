%% dMRI-based BrainAge Model: Transfer Learning Approach
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
x_tune = x_ntuh_af_training; % the input feature for training set
y_tune = y_ntuh_training; % the response for training set
x_test = x_ntuh_af_test; % the input feature for test set
y_test = y_ntuh_test; % the response for test set
x_tune(:,end) = 2; % change the site indicator (default: CamCAN: 1)
x_test(:,end) = 2; % change the site indicator (default: CamCAN: 1)
%% Using the random subset to fine-tune the model
[x_tune,y_tune] = random_select(x_tune,y_tune,200);
% the first to third input argument is 
% the predictors, response, and selected tuning sample size, respectively
fprintf('Random Selection: Done!\n')
%% Load the pretrained model from source domain
% make sure the pretrained model with the same feature types
% CamCAN pretrained model using "tensor" features
% load('Pretrained_Model_CamCAN_usingTensorFeatures.mat')
% CamCAN pretrained model using "advanced" features
load('Pretrained_Model_CamCAN_usingAdvancedFeatures.mat')

%%  Hyperparameter settings:
% 1. layers trainable or not (frozen) 
% 2. regularization (0-1)
% 3. loss function (MSE, MAE, SSE, SAE)

%% Transfer Learning Approach with Random Search (RS)
pt_model = Model_cc; % specify the pre-trained model
max_itr = 300; % specify the max. round of RS

key1 = 0; cp = 0; bestModel = []; bestMAE = Inf;
dim_in = size(x_tune,2);
while key1 == 0
    hp_layerfroz = setfrez(pt_model.numLayers); % 1. layers trainable or not (frozen)
    hp_regur = round(rand(1),2); % 2. regularization (0-1)
    hp_loss = setloss2('Random'); % 3. loss function (MSE, MAE, SSE, SAE)
    
    % 10-fold CV (optional)
    array = CVfold10(x_tune,y_tune);
    model_box = {};
    mae_box = [];
    for fold1 = 1:10
        temp_x_train = array{1,fold1};
        temp_y_train = array{3,fold1};
        temp_x_val = array{2,fold1};
        temp_y_val = array{4,fold1};
        
        %%%   pre-trained model settings  %%%
        % Optimizer
        pt_model.trainFcn = 'traingdx'; % 'trainscg' is faster
        % Epochs
        pt_model.trainParam.epochs = 500;
        % Tolerance of early stopping
        pt_model.trainParam.max_fail = 75;
        % Verbose
        pt_model.trainParam.showWindow = 1;
        % Loss function
        pt_model.performFcn = hp_loss; % mae, mse, sae, sse, etc.
        % Regularization
        pt_model.performParam.regularization = hp_regur; % 0.25, 0.1, 0.01, ect.
        % Freeze partial layers or not (demo)
        fot = logical(hp_layerfroz); % specify which layers should be frozen (frozen: 1).
        % make sure the length of the above vector have the same size with the
        % number of layers in the model (use Model.numLayers to check)
        % [0,0,0,1,1,1] means the fourth to the last layers (close to the input layer) are frozen.
        pt_model = freezelayer(pt_model,fot);
        
        Model_TL = train(pt_model,temp_x_train',temp_y_train','useGPU','yes'); % training
        
        y_hat_temp_val = Model_TL(temp_x_val')';
        mae_val = mean(abs(y_hat_temp_val-temp_y_val));
        model_box{1,fold1} = Model_TL;
        mae_box(1,fold1) = mae_val;
    end
    mae_avg = mean(mae_box);
    [~,loc] = min(abs(mae_box-mae_avg));
    model_sel = model_box{loc};
    if mae_avg < bestMAE
        bestModel = model_sel;
        fprintf('The MAE improved from %g to %g...\n',bestMAE,mae_avg);
        bestMAE = mae_avg;
    else
        fprintf('The MAE did not improve...\n');
    end
    
    cp = cp +1;
    if cp == max_itr
        key1 = 1;
    end
end
fprintf('Random Searching: Done!\n');
fprintf('The "bestModel" is in the workspace\n');

%% Model inference
y_hat_tune = bestModel(x_tune')';
model_inference(y_hat_tune,y_tune,1);

y_hat_test = bestModel(x_test')';
model_inference(y_hat_test,y_test,2);

% if apply the pretrained model directly
y_hat_test = Model_cc(x_test')';
model_inference(y_hat_test,y_test,3);
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


function fot = setfrez(numL)
fot = zeros(1,numL);
for i=1:length(fot)
    k = rand(1);
    if k>=0.75 % prefer to trainable
        fot(i) = 1;
    end
end
k = rand(1);
if k>=0.5
    fot = sort(fot,'descend'); % freeze the layers close to input
else
    fot = sort(fot,'ascend'); % freeze the layers close to output
end
end


function af = setloss2(str)
if strcmp(str,'Random') == 1
    n1 = randperm(4);
    if n1(1) == 1
        af = 'mse';
    elseif n1(1) == 2
        af = 'mae';
    elseif n1(1) == 3
        af = 'sse';
    elseif n1(1) == 4
        af = 'sae';
    end
else
    af = 'mse';
end
end


function array = CVfold10(input,resp)
numo = size(input,1); % number of observations
foldsize = floor(numo/10);
list = randperm(numo);
array = {};
for i=1:10
    temp_list = list;
    val_cor = temp_list((i-1)*foldsize+1:i*foldsize);
    temp_list((i-1)*foldsize+1:i*foldsize) = [];
    tr_cor = temp_list;
    array{1,i} = input(tr_cor,:); % training set
    array{2,i} = input(val_cor,:); % validation set
    array{3,i} = resp(tr_cor); % training set- response
    array{4,i} = resp(val_cor); % validation set- response
    if i == 10
        temp_list = list;
        val_cor = temp_list((i-1)*foldsize+1:numo);
        temp_list((i-1)*foldsize+1:numo) = [];
        tr_cor = temp_list;
        array{1,i} = input(tr_cor,:); % training set
        array{2,i} = input(val_cor,:); % validation set
        array{3,i} = resp(tr_cor); % training set- response
        array{4,i} = resp(val_cor); % validation set- response
    end
end
end
