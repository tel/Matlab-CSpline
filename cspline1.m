function [Mtc, lambda] = cspline1(y)

	%% Constants storing the numerical parameters we're working with.
	%  (precision defines the maximal roundoff error as 2^-PRECISION, bytesize
	%  is the size of the representation.)
	PRECISION = 52;
	BYTESIZE  = 8;

	%% Initial parameters.
	Ny = length(y);
	Nw = Ny-2;
	w  = diff(y, 2);

	%% Fail for really small y b/c we need to make some size assumptions in
	%  order to unroll all of the loops we want to unroll.
	if Nw < 3
		error('cspline1 not implemented for very small y vectors.')
	end

	%% Use L1 cache size to estimate the initial memory allocation size.
	%  (operations are fastest if they (a) do not need to allocate extra
	%  memory and (b) remain in cache. Therefore, even on loops with an
	%  optimally small number of iterations before convergence, we start with
	%  maximally sized arrays that still fit into the cache.)
	CACHE = 4096;
	RESIDE = 0.75; % Only use this percentage of the CACHE
	NARRS = 5;
	Nalloca0 = floor((CACHE*RESIDE/BYTESIZE - Nw)/NARRS);
	if Nalloca0 < 100
		Nalloca0 = 100;
	end

	%% Runtime flags.
	GIVENUP = 0; % Have we exceeded matrix A conditioning yet?

	%% Preallocate the vectors we need to use.
	Nalloca = min([Nalloca0, Nw]);
	f    = zeros(Nalloca, 1);
	e    = zeros(Nalloca, 1);
	th   = zeros(Nw, 1);
	Mtc  = zeros(Ny, 1);

	% Create some endpts for limits of the reconstruction
	Nlim = Nw;

	%% Minimize the GCV via lambda
	opts   = optimset('TolX', 1e-16, 'Display', 'on');
	sigma  = fminbnd(@iterate, 0, 1);
	lambda = sigma^2/(1-sigma^2);

	%% Ensure that the final value of c is using the optimal lambda
	%gcv = iterate(sigma)
	
	%% Store the output in c and return
	Mtc = y - Mtc;

function gcv = iterate(sigma)
	% A single iteration of the smoothing process, computed over the global
	% variables w, e, f, th, and Mtc. Returns the GCV score for this choice of
	% lambda.
	
	%% Decompress lambda.
	lambda = sigma^2/(1-sigma^2);

	%% Check to see if we're ill-conditioned at this point.
	if not(GIVENUP)
		giveupp(Ny, lambda);
	end

	%% Compute the A matrix
	a = 6 + 2/3*lambda;
	b = -4 + lambda/6;

	%% Decide whether we're got enough memory allocated
	empN = Nconverge(lambda);
	if Nalloca < empN
		%% We need more space
		if empN > Nw
			%% We will never converge, but that's ok, we'll just use the max
			Nalloca = Nw;
		else
			%% We will convege before computing the whole vectors still.
			%  Increase the allocated space to cover the convergence through
			%  doubling. Again, limit at Nw.
			Nalloca = min([Nalloca * 2^(ceil(log2(empN) - log2(Nalloca))), Nw]);
		end
		f     = zeros(Nalloca, 1);
		e     = zeros(Nalloca, 1);
		theta = zeros(Nalloca, 1);
	end

	%% FORWARD LOOP:
	%% Compute the e, f, and theta sequences 
	Nlim = min([Nw, empN]);

	%% Unroll the first couple iterations of the loop in order to prime e and f. 
	%  (involves manually inserting zeros for zero'd and negative indicies.)

	% i = 1
	d = a;
	f(1)  = 1/d;
	q     = b;
	e(1)  = -f(1) * q;
	th    = f(1)*w(1);
	theta(1) = th;
	th1 = th;

	% i = 2
	d = (a + q*e(1));
	f(2)  = 1/d;
	qold = q;
	q     = b + e(1);
	e(2)  = -f(2) * q;
	th = f(2)*(w(2) - qold*th1);
	theta(2) = th;
	th2 = th1; th1 = th;

	%% Full loop primed from unrolled values
	for i = 3:Nlim
		d = (a + q*e(i-1) - f(i-2));
		f(i)  = 1/d;
		qold = q;
		q     = b + e(i-1);
		e(i)  = -f(i) * q;
		th = f(i)*(w(i) - th2 - qold*th1);
		theta(i) = th;
		th2 = th1; th1 = th;
	end

	q = b + e(Nlim);
	for i = (Nlim+1):Nw
		th = f(Nlim)*(w(i) - th2 - q*th1);
		theta(i) = th;
		th2 = th1; th1 = th;
	end

	%% BACKWARD LOOP:
	%% Compute the c -> Mtc sequence and the GCV value
	%  Note that this loop must traverse several critical points.
	%  1. Persymmetry: we only need to compute up to the middle for the GCV
	%  2. Limiting: g, h, and q values converge at the same rate as the es and fs
	%  3. Constancy: e, f sequences are constant at the beginning of this
	%     iteration but begin to decrease
	Nodd = mod(Nw, 2) == 1;
	if Nodd
		Nmid = Nw+1 - (Nw-1)/2;
	else
		Nmid = Nw/2+1;
	end
	Nlim_b = Nw-Nlim+1;
	Nback = max([Nmid, Nlim_b]);

	% inits
	tr = 0;
	num = 0;
	
	%% Unroll the first few iterations for the zeros
	% 1.
	i = Nw;
	c = theta(i);
	Mtc(i+2) = c;
	Mtc(i+1) = -2*c;
	Mtc(i)   = c;
	if i >= Nlim
		g = f(Nlim);
	else
		g = f(i);
	end
	h = 0;
	q = 0;

	tr = tr + 6*g;
	num = num + Mtc(i+2)^2;

	c1 = c;
	g1 = g;

	% 2.
	i = Nw-1;
	q(i) = 0;
	if i >= Nlim
		c = theta(i) + e(Nlim) * c1;
		h = e(Nlim) * g1;
		g = f(Nlim) + e(Nlim) * h;
	else
		c = theta(i) + e(i) * c1;
		h = e(i) * g1;
		g = f(i) + e(i) * h;
	end

	Mtc(i+2) = Mtc(i+2) + c;
	Mtc(i+1) = Mtc(i+1) - 2*c;
	Mtc(i)   = c;

	tr = tr + 6*g - 8*h;
	num = num + Mtc(i+2)^2;

	c2 = c1; c1 = c;
	g2 = g1; g1 = g;
	h1 = h;
	
	% Real loop, above the limit
	Nback_lim = max([Nlim, Nback]);
	for i = (Nw-2):-1:Nback_lim
		c = theta(i) + e(Nlim) * c1 - f(Nlim) * c2;
		Mtc(i+2) = Mtc(i+2) + c;
		Mtc(i+1) = Mtc(i+1) - 2*c;
		Mtc(i)   = c;

		q = e(Nlim) * h1 - f(Nlim) * g2;
		h = e(Nlim) * g1 - f(Nlim) * h1;
		g = f(Nlim) * (1 - q) + e(Nlim) * h;

		gcvinc = 6*g - 8*h + 2*q;
		tr = tr + gcvinc;
		num = num + Mtc(i+2)^2;

		c2 = c1; c1 = c;
		g2 = g1; g1 = g;
		h1 = h;
	end

	% Under the limit
	for i = (Nback_lim-1):-1:Nback
		c = theta(i) + e(i) * c1 - f(i) * c2;
		Mtc(i+2) = Mtc(i+2) + c;
		Mtc(i+1) = Mtc(i+1) - 2*c;
		Mtc(i)   = c;

		q = e(i) * h1 - f(i) * g2;
		h = e(i) * g1 - f(i) * h1;
		g = f(i) * (1 - q) + e(i) * h;

		gcvinc = 6*g - 8*h + 2*q;
		tr = tr + gcvinc;
		num = num + Mtc(i+2)^2;

		c2 = c1; c1 = c;
		g2 = g1; g1 = g;
		h1 = h;
	end

	% Finish computing Mtc
	Nback_lim2 = min([Nlim, Nback]);
	for i = (Nback-1):-1:Nback_lim2
		c = theta(i) + e(Nlim) * c1 - f(Nlim) * c2;
		Mtc(i+2) = Mtc(i+2) + c;
		Mtc(i+1) = Mtc(i+1) - 2*c;
		Mtc(i)   = c;

		num = num + Mtc(i+2)^2;

		c2 = c1; c1 = c;
	end
	for i = (Nback_lim2-1):-1:1
		c = theta(i) + e(i) * c1 - f(i) * c2;
		Mtc(i+2) = Mtc(i+2) + c;
		Mtc(i+1) = Mtc(i+1) - 2*c;
		Mtc(i)   = c;

		num = num + Mtc(i+2)^2;

		c2 = c1; c1 = c;
	end

	%% Finalize GCV
	
	% Multiply in the requisite number of limit values
	tr = tr + gcvinc * (Nback - Nmid);
	if Nodd
		% Compute one more value. This will be the limit again if we've
		% reached it.
		q = e(i) * h1 - f(i) * g2;
		h = e(i) * g1 - f(i) * h1;
		g = f(i) * (1 - q) + e(i) * h;

		gcvinc = 6*g - 8*h + 2*q;
		tr = tr*2 + gcvinc;
	else
		% It's just doubled by persymmetry
		tr = tr*2;
	end

	% We also need to add the first two Mtc values which aren't reached in the
	% main loops
	num = num + Mtc(2)^2 + Mtc(1)^2;
	
	% Return the GCV
	gcv = Ny * num / tr^2;

end
	
function n = Nconverge(lambda)
	% Determine the necessary upper bound of iterations required 
	if lambda > 100
		n = 14;
	else
		n = ceil(26.1668 * lambda^(-0.242888));
	end

end

function yn = giveupp(n, l)
	% Estimate the condition of matrix A from N and lambda then emit a warning
	% if we're past numerical precision.
	
	% Compute an estimate of the upper bound of the log condition of A
	llam = log(l);
	ln = log(n);
	if llam > 25.1
		logk = min([1.098612288668110, 0.3983724*llam - 1.1267988]);
	else
		asymptote = ln*3.959716 - 3.272399;
		logk = min([asymptote, -0.9992828 * llam + 2.7798460]);
	end

	% Convert the logk value to base 2
	logk = logk/0.693147180559945;

	if logk > PRECISION
		% We've lost numerical precision. Notify and bail.
		GIVENUP = true;
		fprintf('Problem condition exceeds numerical precision! Answer may be untrustworthy')
	end
	yn = GIVENUP;

end

end
