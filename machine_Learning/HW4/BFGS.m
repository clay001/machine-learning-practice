function H = BFGS(H,sk,yk)
  
  fenmu = 1/(yk'*sk);
  qianfenzi = yk*sk';
  houfenzi = sk*sk';
  E = eye(size(H,1));
  H = (E-fenmu*qianfenzi)*H*(E-fenmu*qianfenzi) + fenmu*houfenzi;
  
end

