function r=Olearning_predict_Single(H1,model)
beta=model{3};
intercept=model{2};
r=sign(intercept+H1*beta);
