close all
clc;clear;
rng default

%% part A
[w1, w2] = A();

%% part B
[m1, m2, s1, s2] = B(w1, w2);

%% part C
C(w1, w2, m1, m2, s1, s2)

%% part D
D(w1, w2)
