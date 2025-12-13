function [lb,ub,D,out] = Func_eng(F)

switch F
    case 'PressureVesselDesign'
        out = @PressureVesselDesign;
        D=4;
        lb=[0.0625 0.0625 10 10];
        ub=[99*0.0625 99*0.0625 200 200];

    case 'SpringDesign'
        out = @SpringDesign;
        D=3;
        lb=[0.05 0.25 2];
        ub=[2 1.3 15];

    case 'ThreeBarTruss'
        out = @ThreeBarTruss;
        D=2;
        lb=[0 0];
        ub=[1 1];

    case 'GearTrainDesign'
        out = @GearTrainDesign;
        D=4;
        lb=[12 12 12 12];
        ub=[60 60 60 60];

    case 'CantileverBeam'
        out = @CantileverBeam;
        D=5;
        lb=[0.01 0.01 0.01 0.01 0.01];
        ub=[100 100 100 100 100];

end
end

function out=PressureVesselDesign(x)

y1=x(:,1);%Ts
y2=x(:,2);%Th
y3=x(:,3);%R
y4=x(:,4);%L
%%% opt
fx=0.6224.*y1.*y3.*y4+...
    1.7781.*y2.*y3.^2+...
    3.1661.*y1.^2.*y4+...
    19.84.*y1.^2.*y3;
%%% const
g(:,1)=-y1+0.0193.*y3;
g(:,2)=-y2+0.0095.*y3;
g(:,3)=-pi.*y3.^2.*y4...
    -(4/3).*pi.*y3.^3 ...
    +1296000;
g(:,4)=y4-240;

%%% Penalty
pp=10^9;
for i=1:size(g,1)
    for j=1:size(g,2)
        if g(i,j)>0
            penalty(i,j)=pp.*g(i,j);
        else
            penalty(i,j)=0;
        end
    end
end

out=fx+sum(penalty,2);

end

function out=SpringDesign(x)

y1=x(:,1);%W
y2=x(:,2);%d
y3=x(:,3);%N
%%% opt
fx=(y3+2).*y2.*y1.^2;
%%% const
g(:,1)=1-(y2.^3.*y3)./(71785.*y1.^4);
g(:,2)=(4.*y2.^2-y1.*y2)./...
    (12566.*(y2.*y1.^3-y1.^4))...
    +(1./(5108.*y1.^2))-1;
g(:,3)=1-(140.45.*y1./(y2.^2.*y3));
g(:,4)=(y1+y2)./1.5-1;
%%% Penalty
pp=10^9;
for i=1:size(g,1)
    for j=1:size(g,2)
        if g(i,j)>0
            penalty(i,j)=pp.*g(i,j);
        else
            penalty(i,j)=0;
        end
    end
end

out=fx+sum(penalty,2);

end

function out=ThreeBarTruss(x)

A1=x(:,1);
A2=x(:,2);
%%%opt
fx=(2*sqrt(2).*A1+A2).*100;
%%% const
g(:,1)=2.*(sqrt(2).*A1+A2)./...
    (sqrt(2).*A1.^2+2.*A1.*A2)-2;
g(:,2)=2.*A2./(sqrt(2).*A1.^2+...
    2.*A1.*A2)-2;
g(:,3)=2./(A1+sqrt(2).*A2)-2;
%%% Penalty
pp=10^9;
for i=1:size(g,1)
    for j=1:size(g,2)
        if g(i,j)>0
            penalty(i,j)=pp.*g(i,j);
        else
            penalty(i,j)=0;
        end
    end
end

out=fx+sum(penalty,2);

end

function out=GearTrainDesign(x)

y1=x(:,1);%A
y2=x(:,2);%B
y3=x(:,3);%C
y4=x(:,4);%D
%%% opt
fx=((1/6.931)-((y1.*y2)./(y3.*y4))).^2;
out=fx;
end

function out=CantileverBeam(x)
y1=x(:,1);%1
y2=x(:,2);%2
y3=x(:,3);%3
y4=x(:,4);%4
y5=x(:,5);%5
%%% opt
fx=0.0624.*(y1+y2+y3+y4+y5);
%%%% const
g(:,1)=(61./y1.^3)+(37./y2.^3)+(19./y3.^3)+(7./y4.^3)+(1./y5.^3)-1;
%%% Penalty
pp=10^9;
for i=1:size(g,1)
    for j=1:size(g,2)
        if g(i,j)>0
            penalty(i,j)=pp.*g(i,j);
        else
            penalty(i,j)=0;
        end
    end
end

out=fx+sum(penalty,2);

end
