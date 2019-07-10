% This is test script for logistic regression 

close all;
clear;
clc;

% create a dataset 
x = [0,0;2,2;2,0;3,0];
y = [0;0;1;1];
c = [1;1;1;1];
x_hom = [c x]; % homogeneous form

[m,n] = size(x);
theta = zeros(size(x_hom,2),1);

[cost,grad,h] = costFunction(theta,x_hom,y);
alpha = 1;
glist = [];
while (norm(grad) > 0.0001)
   [a,b] = size(glist);
   if b < 101
     glist(end+1) = log(norm(grad));
   end
   theta = theta - alpha .* grad;
   [cost,grad,h] = costFunction(theta,x_hom,y);
end
% plot g 
figure(1);
[a,b] = size(glist);
time = 1:1:b;
plot(time,glist);
title('negative gradient');
   
% compute the accuracy of prediction in training set
pred_y = x_hom * theta;
for i = 1:m
  if pred_y(i) < 0
    pred_y(i) = 0;
  end
  if pred_y(i) > 0
    pred_y(i) = 1;
  end
end
count = 0;
for i = 1:m
  if pred_y(i) == y(i)
    count = count+1;
  end
end
ng_pred_accuracy = count/4;
theta
ng_wrong = 4-count

%%newton method----------------------------------------
theta = zeros(size(x_hom,2),1);

[cost,grad,h] = costFunction(theta,x_hom,y);
alpha = 1;
glist = [];
while (norm(grad) > 0.0001)
   [a,b] = size(glist);
   if b < 101
     glist(end+1) = log(norm(grad));
   end
   theta = theta - alpha .* inv(h)* grad;
   [cost,grad,h] = costFunction(theta,x_hom,y);
end
figure(2);
[a,b] = size(glist);
time = 1:1:b;
plot(time,glist);
title('newtons method');

% compute the accuracy of prediction in training set
pred_y = x_hom * theta;
for i = 1:m
  if pred_y(i) < 0
    pred_y(i) = 0;
  end
  if pred_y(i) > 0
    pred_y(i) = 1;
  end
end
count = 0;
for i = 1:m
  if pred_y(i) == y(i)
    count = count+1;
  end
end
newton_pred_accuracy = count/4;
theta
newton_wrong = 4-count

%%BFGS method----------------------------------------
theta = zeros(size(x_hom,2),1);

[cost,grad,h] = costFunction(theta,x_hom,y);
glist = [];
H0 = eye(size(x_hom,2));
w0 = theta;
H = H0;
w = w0; 
kk = 1;
while (norm(grad) > 0.0001) 
   [a,b] = size(glist);
   if b < 101
     glist(end+1) = log(norm(grad));
   end
   if kk == 1
       p = - inv(h)* grad;
   else
       p = - H * grad;
   end
   
   alpha = linesearch(p ,grad,w,x_hom,y);
   w_new = w + alpha*p;
   [cost,grad_new,h_new] = costFunction(w_new,x_hom,y);
   if grad == grad_new
     break
   end
   sk = w_new - w;
   yk = grad_new - grad;
   H = BFGS(H,sk,yk);
   grad = grad_new;
   w = w_new;
end
theta = w;
figure(3);
[a,b] = size(glist);
time = 1:1:b;
plot(time,glist);
title('BFGS method');

% compute the accuracy of prediction in training set
pred_y = x_hom * theta;
for i = 1:m
  if pred_y(i) < 0
    pred_y(i) = 0;
  end
  if pred_y(i) > 0
    pred_y(i) = 1;
  end
end
count = 0;
for i = 1:m
  if pred_y(i) == y(i)
    count = count+1;
  end
end
BFGS_pred_accuracy = count/4;
theta
BFGS_wrong = 4-count


