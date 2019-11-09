%% Brain Age Modeling
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

%% The Demostration of Cascade Neural Network Modeling for Brain Age Regression
% Model configuration and training

% specify the number of layers and the neurons manually
nNeuron = [64,32,16,8,4,2,1]; % number of neurons in the hidden layers

% or
% numL = 5; % number of layer
% nNeuron = round(linspace(size(x_train,2),1,numL+1));
% nNeuron = nNeuron(2:end);

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
CasNN.inputs{1}.size = size(x_train,2);
for i=1:nLay
    CasNN.layers{i}.size = nNeuron(i);
end

% specify the activation function
for i=1:nLay-1
    CasNN.layers{i}.transferFcn = 'logsig'; 
end
CasNN.layers{nLay}.transferFcn = 'poslin';

% min-max normalization
CasNN.inputs{1,1}.processFcns{1,1} = 'mapminmax'; 
% specify the optimizer
CasNN.trainFcn = 'traingdx';
% specify verbose logging or not
CasNN.trainParam.showWindow = 1;
% specify the loss function
CasNN.performFcn = 'mse';
% specify error normalization
CasNN.performParam.normalization = 'standard';
% specify number of epoch
CasNN.trainParam.epochs = 1000; 
CasNN.plotFcns = {'plotperform'};
% specify the proportion of hold-out validatoin for early stopping
CasNN.divideFcn = 'dividerand'; 
CasNN.divideParam.testRatio = 0;
CasNN.divideParam.trainRatio = 0.85;
CasNN.divideParam.valRatio = 0.15;
% specify the degree of regularization
CasNN.performParam.regularization = 0.65;
% specify the early stopping criterion
CasNN.trainParam.max_fail = 100;

% configure and train
CasNN = configure(CasNN,x_train',y_train');
[Model1,train_rec] = train(CasNN,x_train',y_train','useGPU','yes');
fprintf('Done!\n')

%% CasNN model inference
y_hat_train = Model1(x_train')';
model_inference(y_hat_train,y_train,1);

y_hat_test = Model1(x_test')';
model_inference(y_hat_test,y_test,2);

%% Function
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

