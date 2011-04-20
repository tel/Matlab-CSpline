N = 10000;
dc = 5;

ts = (1:N)/N;
pure = sin(ts*1000*pi);

noise = randn(1,N);
ac = zeros(1,N);
ac(1) = noise(1);
corr = 0.2;
for i = 2:N
    ac(i) = ac(i-1)*corr + noise(i);
end
ac = ac+dc;