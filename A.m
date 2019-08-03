function [ w1, w2 ] = A(  )
%Part A: Generate data

%% part A
fprintf ('part A: Generate data\n\n')

%w1 class
a = 2; 
b = 8;
c = 1; 
d = 2;
x1 = a + (b-a).*rand(400,1);
x2 = c + (d-c).*rand(400,1);

w1 = [x1, x2];

%w2 class
a = 6; 
b = 8;
c = 2.5; 
d = 5.5;
x1 = a + (b-a).*rand(100,1);
x2 = c + (d-c).*rand(100,1);

w2 = [x1, x2];

%plot the 2 classes
hold on
figure(1)
scatter(w1(:,1), w1(:,2), 'b*');
scatter(w2(:,1), w2(:,2), 'ro');

xlim([-1 9])
ylim([-1 7])
% Set the axes to go through the origin
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('2-D Space with our samples');
legend('w1','w2','Location','northwest');
hold off

end