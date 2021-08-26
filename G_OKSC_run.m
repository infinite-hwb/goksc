clc
clear, close all
addpath('.\performance');
addpath('.\data');
addpath('.\tool');
%% Load Dataset
load yale_GLRR;
dataset = 'yale_GLRR';     
K=GTR;
gnd=gnd;
eta = 20;
[~,N]=size(K);
F = randn(N,N);    
F= orth(F);
c=max(gnd);
%% G_OKSC
% loss function parameters
lambda1=120;
lambda2=0.5;
lambda3=150;
beta=0.002;
% One-step get cluster indicator matrix
[F] = OKSC(K,F,c,lambda1,lambda2, lambda3, beta, eta);
% The average performance of 20 kmeans
for q=1:20
    grp = kmeans(F, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
    CAq(q) = 1-compute_CE(grp, gnd);          % clustering accuracy
    [Fq(q),Pq(q),Rq(q)] = compute_f(gnd,grp); % F1, precision, recall
    nmiq(q) = compute_nmi(gnd,grp);           % NMI performance
    ARq(q) = rand_index(gnd,grp);             % ARI performance
end
% average clustering performance with standard deviation
CA(1)=mean(CAq);     CA(2)=std(CAq);
Fscore(1)=mean(Fq);  Fscore(2)=std(Fq);
Pscore(1)=mean(Pq);  Pscore(2)=std(Pq);
Rscore(1)=mean(Rq); Rscore(2)=std(Rq);
nmi(1)=mean(nmiq);    nmi(2)=std(nmiq);
AR(1)=mean(ARq);     AR(2)=std(ARq);
avg.CA=CA;
avg.F=Fscore;
avg.P=Pscore;
avg.R=Rscore;
avg.nmi=nmi;
avg.AR=AR;
% show avg
avg