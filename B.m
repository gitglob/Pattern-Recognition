function [ m1, m2, s1, s2 ] = B( w1, w2 )
%Part B: Bayessian Classification in 2-D space
fprintf ('part B: Bayessian Classification in 2-D space\n\n')

%% B.1

% find mean values of samples
m1 = mean(w1);
m2 = mean(w2);
disp(['w1 mean is [',num2str(m1),']']);
disp(['w2 mean is [',num2str(m2),']',]);

% find covariances
s1 = cov(w1);
s2 = cov(w2);

% calculate the posterior probability distributions
p11 = zeros(400,1);
p12 = zeros(400,1);
for i = 1:400
    p11(i) = (1/(2*pi*sqrt(det(s1)))) * exp(-(1/2)*(w1(i,:) - m1)*inv(s1)*(w1(i,:) - m1)');
    p12(i) = (1/(2*pi*sqrt(det(s2)))) * exp(-(1/2)*(w1(i,:) - m2)*inv(s2)*(w1(i,:) - m2)');
    
end

p21 = [];
p22 = [];
for i = 1:100
    p22 = [p22,(1/(2*pi*sqrt(abs(det(s2))))) * exp(-(1/2)*(w2(i,:) - m2)*inv(s2)*(w2(i,:) - m2)')];
    p21 = [p21,(1/(2*pi*sqrt(abs(det(s1))))) * exp(-(1/2)*(w2(i,:) - m1)*inv(s1)*(w2(i,:) - m1)')];
end

% plot each instance with its posterior probability
figure(2)
stem3(w1(:,1),w1(:,2),p11');
title('w1 instances and posterior probability')

figure(3)
stem3(w2(:,1),w2(:,2),p22');
title('w2 instances and posterior probability')

fprintf ('\n')

%% B2

w1r = [];
w2r = [];
w1w = [];
w2w = [];

% classify according to the euclidean norm
for i = 1:400
    % euclidian distance
    d1 = norm(w1(i,:) - m1);
    d2 = norm(w1(i,:) - m2);
    if d1 <= d2 % correct prediction
        w1r = [w1r;w1(i,:)];
    else %wrong prediction
        w1w = [w1w;w1(i,:)];
    end
end
for i = 1:100
    d1 = norm(w2(i,:) - m1);
    d2 = norm(w2(i,:) - m2);
    if d1 < d2 % wrong prediction
        w2w = [w2w;w2(i,:)];
    else % right prediction
        w2r = [w2r;w2(i,:)];
    end
end

% calculate error %
error = (size(w1w,1) + size(w2w,1)) / 500;
disp(['NN (Euc) error : ',num2str(error*100),'%']);

% plot missclassified samples if any
figure(4)

hold on;
plot(w1(:,1),w1(:,2),'bo',w2(:,1),w2(:,2),'gs');

if size(w1w,1) ~= 0 
    plot(w1w(:,1),w1w(:,2),'r*');
end
if size(w2w,1) ~= 0
    plot(w2w(:,1),w2w(:,2),'k*');
end

%% B3

% classify according to the Mahalanobis Distance
s = (s1+s2)/2;
w1r = [];
w2r = [];
w1w = [];
w2w = [];
for i = 1:400
    % Mahalanobis Distance
    d1 = sqrt((w1(i,:) - m1)*inv(s)*(w1(i,:) - m1)');
    d2 = sqrt((w1(i,:) - m2)*inv(s)*(w1(i,:) - m2)');
    if d1 <= d2 % correct predictions
        w1r = [w1r;w1(i,:)];
    else % wrong predictions
        w1w = [w1w;w1(i,:)];
    end
end
for i = 1:100
    d1 = sqrt((w2(i,:) - m1)*inv(s)*(w2(i,:) - m1)');
    d2 = sqrt((w2(i,:) - m2)*inv(s)*(w2(i,:) - m2)');
    if d1 < d2 % wrong predictions
        w2w = [w2w;w2(i,:)];
    else % right predictions
        w2r = [w2r;w2(i,:)];
    end
end

% calculate error %
error = (size(w1w,1) + size(w2w,1)) / 500;
disp(['NN (Mah) error : ',num2str(error*100),'%']);

% plot missclassified samples if any
if size(w1w,1) ~= 0 
    % we have no wrong classification in class 1, so this will produce a
    % useless warning
    plot(w1w(:,1),w1w(:,2),'c^');
end
if size(w2w,1) ~= 0
    plot(w2w(:,1),w2w(:,2),'m^');
end

xlim([-1 9])
ylim([-1 7])
% Set the axes to go through the origin
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('2-D Sample Plot with errors');
legend('w1 Sample','w2 Sample','Euc. errors','Euc. errors','Mah. errors','Mah. errors','Location','northwest');

hold off

%% B4

% classify according to the posterior probabilities
w1r = [];
w2r = [];
w1w = [];
w2w = [];
for i = 1:400
    if p11 >= p12 % rightfully classify to class 1
        w1r = [w1r;w1(i,:)];
    else % wrongfully classify to class 1
        w1w = [w1w;w1(i,:)];
    end
end
for i = 1:100
    if p21 > p22 % wrongfully classify to class 2
        w2w = [w2w;w2(i,:)];
    else % rightfully classify to class 2
        w2r = [w2r;w2(i,:)];
    end
end

% calculate error percentage for both classes
error = (size(w1w,2) + size(w2w,2)) / 500;
fprintf ('\n')
disp(['Bayesian error : ',num2str(error*100),'%']);

end

