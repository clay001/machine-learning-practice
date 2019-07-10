close all;
clear;
clc;

% nba_data is a matrix 
% each row denotes the performance of a certain NBA team, which contains 
% scores, assits, backboard, etc. The last column indicate the team win 
% champions or not
load nbadata;
x = nba_data(:,1:22);
y = nba_data(:,23);
c = ones(size(y));
x_hom = [c x]; % homogeneous form
[m,n] = size(x);

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

BFGS_before_pred_accuracy = count/size(y,1)
BFGS_before_wrong = size(y,1)-count;

