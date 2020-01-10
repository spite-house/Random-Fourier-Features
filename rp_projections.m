function [W, B] = rp_projections(D,d,kernel)
switch kernel
    case 'gaussian'
        W = sqrt(2/double(d))*randn(d,D);
        B = 2*pi*rand(d,1);
    %case 'linear'
    %    W = sqrt(2)*rand(d,D)/sqrt(D);   
    otherwise
        error('cannot sample from that yet.')
end
end
