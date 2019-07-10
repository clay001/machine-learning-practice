function alpha = linesearch(p,grad,theta,x,y)
   alpha = 1;
   rou = 0.45;
   c = 0.2;
   move = theta + alpha*p;
   [cost1,fei1,fei2] = costFunction(theta,x,y);
   [cost2,fei1,fei2] = costFunction(move,x,y);
   while(cost2 > (cost1 + c*alpha*grad'*p))
       alpha = rou*alpha;
       move = theta + alpha*p;
       [cost2,fei1,fei2] = costFunction(move,x,y);
   end

end
  
