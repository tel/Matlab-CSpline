function y = jump_signal()

N = 10000;
ts = (1:N)/N;
ys = sin(pi*ts*5000) + cos(pi*ts*200);

Q = 0.1;
v = mod(ts, Q) > Q/2;
v = randn(1, N).*v;

y = ys + 0.5*v.*ys;


y = y';
end