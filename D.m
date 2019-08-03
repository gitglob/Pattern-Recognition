function [  ] = D( w1, w2 )
%Part D: Linear classification with several cost functions
fprintf ('\npart D: Linear classification with several cost functions\n\n')

%% D1 Least Squares
y = ones(500,1);
y(401:500) = -y(401:500);
X = ones(500,3); % col for stable coefficients
X(:,1:2) = [w1;w2];
w = inv(X'*X)*X'*y;

figure(13)
plot(w1(:,1),w1(:,2),'bp',w2(:,1),w2(:,2),'gh');
hold on;
syms x y;
f(x,y) = w(1)*x + w(2)*y + w(3);
p = ezplot(f,[0,10]);
set(p, 'Color', 'm','DisplayName','classifier');
legend ('w1', 'w2', 'classifier')
title('Least Squares');

% calculate LSE
J = zeros(500,1);
y = ones(500,1);
y(401:500) = -y(401:500);
for i = 1:500
    y_true = y(i);
    y_pred(i) = X(i,:)*w;
    J(i) = (y_true - y_pred(i))^2;
end
lse = sum(J);
disp(['Minimum least Square Error is: ',num2str(lse)]);


% find the missclassifications
errors = 0;
for i = 1:400
    if y_pred(i) < 0
        errors = errors +1;
    end
end
for  i = 401:500
    if y_pred(i) >= 0 
        errors = errors +1;
    end
end

% calculate error percentage for both classes
error = errors / 500;
disp(['Least Squares classification error : ',num2str(error*100),'%']);

%% D2 Perceptron

% initialize weights and learning rate
w = [0; -1; 1];
r = 0.25;

% initialize our variables and our class variable
y = ones(500,1);
y(401:500) = -y(401:500);
X = [w1;w2];
X = [X,ones(500,1)];

% the perceptron algorithm
Y = []; 
steps = 0;
t = 0;
a = 1;
while length(Y)|| a ~= 0
    steps = steps + 1;
    a = 0;
    Y = [];
    for i = 1:500
        if i<=400
            if X(i,:)*w<0
                Y = [Y;1*X(i,:)];
            end
        else
            if X(i,:)*w>=0
                Y = [Y;(-1)*X(i,:)];
            end
        end
    end
    t = t +1;
    sm = [];
    for i = 1:size(Y,1)
      sm = [sm ;Y(i,:)];
    end
    a = sum(sm);
    w = w + r*a';
end

fprintf ('\n')
disp(['finished in: ',num2str(steps),' steps']);

% plot original data and classifier
figure(14)
hold on;
plot(w1(:,1),w1(:,2),'bp',w2(:,1),w2(:,2),'gh');

xx = linspace(0,10,1000);
plot(xx,(xx*w(1)+w(3))/(-w(2)))
legend('w1','w2','classifier');
title('Perceptron');

hold off

% calculate classification errors
x = [w1; w2];
errors = 0;
for i = 1:400
    pred =  w(1)*x(i,1) + w(2)*x(i,2) + w(3);  
    if pred < 0
        errors = errors + 1;
    end
end
for i = 401:500
    pred =  w(1)*x(i,1) + w(2)*x(i,2) + w(3); 
    if pred >= 0
        errors = errors + 1;
    end
end

% calculate error percentage for both classes
error = errors / 500;
disp(['Perceptron classification error : ',num2str(error*100),'%']);


fprintf ('\n\tThe end :) \n')
end