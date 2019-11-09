%% Random Searching for Hyperparameters Tunning in Brain Age Modeling
% Developed by C.L. Chen
%% Load the tensor-based features
load('Data_proc_tensorfeature_CamCAN_Train_Test_N500_N116.mat')
fprintf('The data are allocated!\n');
%% Load the advanced features
load('Data_proc_advfeature_CamCAN_Train_Test_N500_N116.mat')
fprintf('The data are allocated!\n');
%% Specify the features and responses for the training and test sets 
x_train = x_cc_af_training; % the input feature for training set
y_train = y_cc_training; % the response for training set
x_test = x_cc_af_test; % the input feature for test set
y_test = y_cc_test; % the response for test set
%%  Hyperparameter settings:
% 1. # of hidden layers
% 2. # of neurons in hidden layers (2^n series as default setting)
% 3. regularization
% 4. activation function b/w hidden layers

%% Brain Age Modeling with Random Search (RS)

max_itr = 300; % specify the max. round of RS

key1 = 0; cp = 0; bestModel = []; bestMAE = Inf;
dim_in = size(x_train,2);
while key1 == 0
    hp_numL = round(rand(1)*8)+2; % 1. # of hidden layers
    hp_numN = setneuro2(hp_numL,dim_in); % 2. # of neurons in hidden layers (2^n series as default setting)
    hp_regur = round(rand(1),2); % 3. regularization
    hp_af = setaf2('Random'); % 4. activation function b/w hidden layers
    
    % 10-fold CV
    array = CVfold10(x_train,y_train);
    model_box = {};
    mae_box = [];
    for fold1 = 1:10
        temp_x_train = array{1,fold1};
        temp_y_train = array{3,fold1};
        temp_x_val = array{2,fold1};
        temp_y_val = array{4,fold1};
        
        numL = hp_numL;
        nNeuron = [hp_numN,1];
        nLay = length(nNeuron);
        CasNN = feedforwardnet;
        CasNN.numInputs = 1;
        CasNN.numLayers = nLay;
        CasNN.biasConnect = ones(1,nLay)';
        CasNN.inputConnect = ones(1,nLay)';
        CasNN.outputConnect = [zeros(1,nLay-1),1];
        CasNN.layerConnect = tril(ones(nLay),-1);
        CasNN.name = 'CasNN';
        for i = 1:nLay
            CasNN.layers{i}.name = ['L',num2str(i)];
        end
        CasNN.inputs{1}.size = size(temp_x_train,2);
        for i=1:nLay
            CasNN.layers{i}.size = nNeuron(i);
        end
        % specify the activation function
        for i=1:nLay-1
            CasNN.layers{i}.transferFcn = hp_af;
        end
        % the activation function of the last layer, use 'poslin'
        CasNN.layers{nLay}.transferFcn = 'poslin';
        % min-max normalization
        CasNN.inputs{1,1}.processFcns{1,1} = 'mapminmax';
        % specify the optimizer
        CasNN.trainFcn = 'traingdx'; % 'trainscg' is faster
        % specify verbose logging or not
        CasNN.trainParam.showWindow = 0;
        % specify the loss function
        CasNN.performFcn = 'mse';
        % specify error normalization
        CasNN.performParam.normalization = 'standard';
        % specify number of epoch
        CasNN.trainParam.epochs = 800;
        CasNN.plotFcns = {'plotperform'};
        % specify the proportion of hold-out validatoin for early stopping
        CasNN.divideFcn = 'dividerand';
        CasNN.divideParam.testRatio = 0;
        CasNN.divideParam.trainRatio = 0.85;
        CasNN.divideParam.valRatio = 0.15;
        % specify the degree of regularization
        CasNN.performParam.regularization = hp_regur;
        % specify the early stopping criterion
        CasNN.trainParam.max_fail = 100;
        % configure and train
        CasNN = configure(CasNN,temp_x_train',temp_y_train');
        Model1 = train(CasNN,temp_x_train',temp_y_train','useGPU','yes');
        y_hat_temp_val = Model1(temp_x_val')';
        mae_val = mean(abs(y_hat_temp_val-temp_y_val));
        model_box{1,fold1} = Model1;
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
%% CasNN model inference
y_hat_train = bestModel(x_train')';
model_inference(y_hat_train,y_train,1);

y_hat_test = bestModel(x_test')';
model_inference(y_hat_test,y_test,2);

%% Function

function numN = setneuro2(numL,dim_in)
numN = zeros(1,numL);
sup = floor(log2(dim_in));
for i=1:length(numN)
    x = randperm(sup);
    numN(1,i) = 2^x(1);
end
numN = sort(numN,'descend');
end


function af = setaf2(str)
if strcmp(str,'Random') == 1
    n1 = randperm(4);
    if n1(1) == 1
        af = 'logsig';
    elseif n1(1) == 2
        af = 'tansig';
    elseif n1(1) == 3
        af = 'satlin';
    elseif n1(1) == 4
        af = 'poslin';
    end
else
    af = 'logsig';
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