function kind = wcH2( X,TH )
%h2弱分类器
X1=X(1);
X2=X(2);
if X1<TH
    kind=1;
else
    kind=-1;
end
end

