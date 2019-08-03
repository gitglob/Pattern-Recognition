function [  ] = C( w1, w2, m1, m2, s1, s2  )
%Part C: Features' dimensionality deduction
fprintf ('\npart C: Features'' dimensionality deduction\n\n')

%% C1 PCA Feature Reduction

x = [w1;w2];

% De-mean (MATLAB will de-mean inside of PCA, but I want the de-meaned values later)
x = bsxfun(@minus,x,mean(x));

% returns the principal component coefficients, score, variances
[coeff, score, latent, ~, explained] = pca(x); 

% variance explained by PC1 (same as explained[1])
tot_var = latent(1)/sum(latent);
disp(['Variance explained by PC1 : ',num2str(tot_var*100),'%']);

% function to create pc1, pc2
syms a b;
f1(a,b) = coeff(1,1)*a + coeff(2,1)*b;
f2(a,b) = coeff(1,2)*a + coeff(2,2)*b;

% Multiply the original data by the principal component vectors to get 
% the projections of the original data on the principal component vector space (score).
G = x * coeff;

figure (5)
hold on;
plot(w1(:,1),w1(:,2),'g.');
plot(w2(:,1),w2(:,2),'k.');
plot(x(1:400,1), x(1:400,2), 'b*', 'MarkerSize',2);
plot(x(401:500,1), x(401:500,2), 'ro', 'MarkerSize',2);
plot(score(1:400,1), score(1:400,2), 'm^', 'MarkerSize',2);
plot(score(401:500,1), score(401:500,2), 'c<', 'MarkerSize',2);

p1 = ezplot(f1,[-6,6]);
p2 = ezplot(f2,[-6,6]);
set(p1, 'Color', 'g');
set(p2, 'Color', 'k');

xlim([-5 10])
ylim([-5 10])
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('PCA');
xlabel('x')
ylabel('y')
legend('original w1', 'original w2', 'centered w1', 'centered w2',...
    'projected w1','projected w2','PC1', 'PC2');
% the projections are to a different set of coordinates!!
hold off

figure(6)
% Visualize both the orthonormal principal component coefficients for each 
% variable and the principal component scores for each observation in a single plot.
biplot(coeff(:,1:2),'scores',score(:,1:2),...
    'varlabels',{'w1 contribution','w2 contribution'} );
xlabel('PC1')
ylabel('PC2')
title ('PCA biplot')


%% C2 NN_PCA

w1w = [];
w1r = [];
w2w = [];
w2r = [];

% new means at the projected data
m1 = mean([G(1:400,1) G(1:400,2)]) ;
m2 = mean([G(401:500,1) G(401:500,2)]) ;

% first class
for i = 1:400
   % use euclidean norm
   d1 = norm(G(i,:) - m1);
   d2 = norm(G(i,:) - m2);
   if d1 <= d2 % correct predictions
        w1r = [w1r;G(i,:)];
   else % wrong predictions
        w1w = [w1w;G(i,:)];
   end
end
% second class
for i = 401:500
   % use euclidean norm
   d1 = norm(G(i,:) - m1);
   d2 = norm(G(i,:) - m2);
   if d1 < d2 % wrong predictions
       w2w = [w2w;G(i,:)];
   else % correct predictions
       w2r = [w2r;G(i,:)];
   end
end

figure(7)
hold on;
plot(G(1:400,1),G(1:400,2),'bo',G(401:500,1),G(401:500,2),'gs');

if size(w1w,1) ~= 0 
    plot(w1w(:,1),w1w(:,2),'k*');
end
if size(w2w,1) ~= 0
    plot(w2w(:,1),w2w(:,2),'r*');
end
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('PCA errors');
legend('w1 projected Sample','w2 projected Sample',...
    'PCA euc. errors','Location','northwest');
hold off

% calculate error percentage for both classes
error = (size(w1w,1) + size(w2w,1)) / 500;
disp(['After PCA (Euc) classification error with 2-D data : ',num2str(error*100),'%']);

x = [w1;w2];
x = bsxfun(@minus,x,mean(x));

% Now let's classify with the 1-D projections 

% create our PC1 vector
t1 = -(coeff(1,1)*-5)/coeff(2,1) ;
t2 = -(coeff(1,1)*5)/coeff(2,1) ;
v1 = [-5, t1];
v2 = [5, t2];
v = [v1; v2]; 

% calculate the projections to PC1
projection = zeros(500,2);
for i = 1:500
    projection(i,:) = proj(v, x(i,:));
end

figure(8)
hold on
plot(x(1:400,1),x(1:400,2),'g.');
plot(x(401:500,1),x(401:500,2),'b.');
plot(projection(1:400,1),projection(1:400,2),'go',...
    projection(401:500,1),projection(401:500,2),'bs');

p1 = ezplot(f1,[-6,6]);
set(p1, 'Color', 'm');

set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title ('PCA 1-D projections classified')
legend('w1','w2','w1 projections','w2 projections','PC1');
hold off

% now let's check the classification accuracy
w1w = [];
w1r = [];
w2w = [];
w2r = [];

% new means at the projected data
m1 = mean([projection(1:400,1) projection(1:400,2)]) ;
m2 = mean([projection(401:500,1) projection(401:500,2)]) ;

% first class
for i = 1:400
   % use euclidean norm
   d1 = norm(projection(i,:) - m1);
   d2 = norm(projection(i,:) - m2);
   if d1 <= d2 % correct predictions
        w1r = [w1r; projection(i,:)];
   else % wrong predictions
        w1w = [w1w; projection(i,:)];
   end
end
% second class
for i = 401:500
   % use euclidean norm
   d1 = norm(projection(i,:) - m1);
   d2 = norm(projection(i,:) - m2);
   if d1 < d2 % wrong predictions
       w2w = [w2w; projection(i,:)];
   else % correct predictions
       w2r = [w2r; projection(i,:)];
   end
end

figure(9)
hold on;
plot(projection(1:400,1),projection(1:400,2),'bo',...
    projection(401:500,1),projection(401:500,2),'gs');

p1 = ezplot(f1,[-6,6]);
set(p1, 'Color', 'm','DisplayName','PC1');

if size(w1w,1) ~= 0 
    plot(w1w(:,1),w1w(:,2),'k*');
end
if size(w2w,1) ~= 0
    plot(w2w(:,1),w2w(:,2),'r*');
end
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('PCA errors');
legend('w1 projected','w2 projected', 'LDA classifier',...
    'PCA euc. errors w1','PCA euc. errors w2','Location','northwest');
hold off

% calculate error percentage for both classes
error = (size(w1w,1) + size(w2w,1)) / 500;
disp(['PCA 1-D projections (Euc) classification error : ',num2str(error*100),'%']);



%% C3 LDA Featsure Reduction
% LDA is like PCA, but it focuses on maximizing the seprability among known
% categories

% LDA creates a new axis and it projects the data onto that new axis to
% maximize the sepration between the 2 categories (at least at 2D)

% our data
x = [w1;w2];

% make a 'class' variable
y = [zeros(400,1); ones(100,1)];

% classify with LDA
lda = fitcdiscr(x,y);
ldaClass = predict(lda, x(:,:));

% missclassification error
error = resubLoss(lda);

% calculate error percentage for both classes
fprintf ('\n')
disp(['LDA 2-D classification error : ',num2str(error*100),'%']);

figure(10)
% count and plot how many wrong classifications we had
bad = ~ldaClass == y;
sum(bad(:) == 1);
hold on;

gscatter(x(:,1), x(:,2), y, 'rgb','osd');
plot(x(bad,1), x(bad,2), 'k*');

set(gca,'XAxisLocation','origin','YAxisLocation','origin')
xlabel('x')
ylabel('y')
legend('w1','w2','missclassified instances')
title('LDA classification errors')
hold off;

K = lda.Coeffs(1,2).Const ;
L = lda.Coeffs(1,2).Linear ;


% Function to compute K + L*v for multiple vectors
% v=[x;y]. Accepts x and y as scalars or column vectors. 
syms x1 x2;
f(x1,x2) = K + L(1)*x1 + L(2)*x2 ;

% calculate 2 spots in our classifier and its vector
t1 = (-K - L(1)*0)/L(2);
t2 = (-K - L(1)*8.5)/L(2);
v1 = [0,t1];
v2 = [8.5,t2];
v = [v1;v2];

projection = zeros(500,2);
for i = 1:500
    projection(i,:) = proj(v, x(i,:));
end

figure (11)
hold on
plot(x(1:400,1),x(1:400,2),'g.');
plot(x(401:500,1),x(401:500,2),'k.');
plot(projection(1:400,1),projection(1:400,2),'g*');
plot(projection(401:500,1),projection(401:500,2),'ko');

h2 = ezplot(f,[1 8, 0 6]);
set(h2, 'Color', 'm','DisplayName','Decision Boundary')

axis([1 8 0 6])
xlabel('x')
ylabel('y')
legend('w1','w2','w1 projections','w2 projections','classifier')
title('LDA classifier')
hold off


%% C4
% now let's check how our classification would work if we were only
% depnding on the 1-d projections on our LDA and the euclidean norm

w1w = [];
w1r = [];
w2w = [];
w2r = [];

% new means at the projected data
m1 = mean([projection(1:400,1) projection(1:400,2)]) ;
m2 = mean([projection(401:500,1) projection(401:500,2)]) ;

% first class
for i = 1:400
   % use euclidean norm
   d1 = norm(projection(i,:) - m1);
   d2 = norm(projection(i,:) - m2);
   if d1 <= d2 % correct predictions
        w1r = [w1r; projection(i,:)];
   else % wrong predictions
        w1w = [w1w; projection(i,:)];
   end
end
% second class
for i = 401:500
   % use euclidean norm
   d1 = norm(projection(i,:) - m1);
   d2 = norm(projection(i,:) - m2);
   if d1 < d2 % wrong predictions
       w2w = [w2w; projection(i,:)];
   else % correct predictions
       w2r = [w2r; projection(i,:)];
   end
end

figure(12)
hold on;
plot(projection(1:400,1),projection(1:400,2),'bo',...
    projection(401:500,1),projection(401:500,2),'gs');

h2 = ezplot(f,[1 8, 0 6]);
set(h2, 'Color', 'm','DisplayName','Decision Boundary')


if size(w1w,1) ~= 0 
    plot(w1w(:,1),w1w(:,2),'k*');
end
if size(w2w,1) ~= 0
    plot(w2w(:,1),w2w(:,2),'r*');
end
set(gca,'XAxisLocation','origin','YAxisLocation','origin')
title('LDA errors');
legend('w1 projected','w2 projected', 'LDA classifier',...
    'LDA euc. errors w1','LDA euc. errors w2','Location','northwest');
hold off

% calculate error percentage for both classes
error = (size(w1w,1) + size(w2w,1)) / 500;
disp(['LDA 1-D projections (Euc) classification error : ',num2str(error*100),'%']);

end

