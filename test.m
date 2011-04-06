function t = test(y, x, lam)

	[x1, lam1] = cspline1(y);

	e = x - x1;
	err = max(abs(e./x))
	lamerr = abs(lam - lam1)/lam
	clf;
	plot(e(1:100))
