function [lowerbound,upperbound,dimension,fitness] = CEC2017(F)


switch F
    %CEC2017 - 1 
    case 'F1'
        fitness = @F1;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 2
    case 'F2'
        fitness = @F2;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 3
    case 'F3'
        fitness = @F3;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 4
    case 'F4'
        fitness = @F4;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 5
    case 'F5'
        fitness = @F5;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 6
    case 'F6'
        fitness = @F6;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 7
    case 'F7'
        fitness = @F7;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 8
    case 'F8'
        fitness = @F8;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 9
    case 'F9'
        fitness = @F9;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 10
    case 'F10'
        fitness = @F10;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 11
    case 'F11'
        fitness = @F11;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 12
    case 'F12'
        fitness = @F12;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 13
    case 'F13'
        fitness = @F13;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 14
    case 'F14'
        fitness = @F14;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 15
    case 'F15'
        fitness = @F15;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 16
    case 'F16'
        fitness = @F16;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 17
    case 'F17'
        fitness = @F17;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 18
    case 'F18'
        fitness = @F18;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 19
    case 'F19'
        fitness = @F19;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 20
    case 'F20'
        fitness = @F20;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 21
    case 'F21'
        fitness = @F21;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 22
    case 'F22'
        fitness = @F22;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 23
    case 'F23'
        fitness = @F23;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 24
    case 'F24'
        fitness = @F24;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 25
    case 'F25'
        fitness = @F25;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 26
    case 'F26'
        fitness = @F26;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 27
    case 'F27'
        fitness = @F27;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 28
    case 'F28'
        fitness = @F28;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 29
    case 'F29'
        fitness = @F29;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
        
    %CEC2017 - 30
    case 'F30'
        fitness = @F30;
        lowerbound=-100;
        upperbound=100;
        dimension=10;
end

end

% F1
function o = F1(x) 
    o = cec17_func(x',1);
end

% F2
function o = F2(x) 
    o = cec17_func(x',2);
end

% F3
function o = F3(x) 
    o = cec17_func(x',3);
end

% F4
function o = F4(x) 
    o = cec17_func(x',4);
end

% F5
function o = F5(x) 
    o = cec17_func(x',5);
end

% F6
function o = F6(x) 
    o = cec17_func(x',6);
end

% F7
function o = F7(x) 
    o = cec17_func(x',7);
end

% F8
function o = F8(x) 
    o = cec17_func(x',8);
end

% F9
function o = F9(x) 
    o = cec17_func(x',9);
end

% F10
function o = F10(x) 
    o = cec17_func(x',10);
end


% F11
function o = F11(x) 
    o = cec17_func(x',11);
end

% F12
function o = F12(x) 
    o = cec17_func(x',12);
end


% F13
function o = F13(x) 
    o = cec17_func(x',13);
end

% F14
function o = F14(x) 
    o = cec17_func(x',14);
end

% F15
function o = F15(x) 
    o = cec17_func(x',15);
end

% F16
function o = F16(x) 
    o = cec17_func(x',16);
end


% F17
function o = F17(x) 
    o = cec17_func(x',17);
end

% F18
function o = F18(x) 
    o = cec17_func(x',18);
end

% F19
function o = F19(x) 
    o = cec17_func(x',19);
end

% F20
function o = F20(x) 
    o = cec17_func(x',20);
end

% F21
function o = F21(x) 
    o = cec17_func(x',21);
end

% F22
function o = F22(x) 
    o = cec17_func(x',22);
end

% F23
function o = F23(x) 
    o = cec17_func(x',23);
end

% F24
function o = F24(x) 
    o = cec17_func(x',24);
end


% F25
function o = F25(x) 
    o = cec17_func(x',25);
end

% F26
function o = F26(x) 
    o = cec17_func(x',26);
end

% F27
function o = F27(x) 
    o = cec17_func(x',27);
end

% F28
function o = F28(x) 
    o = cec17_func(x',28);
end

% F29
function o = F29(x) 
    o = cec17_func(x',29);
end

% F30
function o = F30(x) 
    o = cec17_func(x',30);
end
