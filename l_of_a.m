function [e, f] = l_of_a(lambda, N)

% Initialize the parameters of A
a = 6 + 2/3*lambda;
b = -4 + lambda/6;

% Allocate storage for the L-sequences, including the bottom-out zero
% padding.  
e = zeros(1, N-3+2);
f = zeros(1, N-4+2);

% Determine the necessary upper bound of iterations required
% This is either the midpoint of the sequence (abusing persymmetry) 
%   or the convergent limit, empirically estimated.
rho = lambda;
if lambda > 100
  empiricalN = 14;
else
  empiricalN = ceil(26.1668 * lambda^(-0.242888));
end

Nend = min(N, empiricalN);

% Run the iteration, filling e and f
d = 0;
q = b;
for i = (1+2):(Nend+2)
	d     = (a + q*e(i-1) - f(i-2));
	f(i)  = 1/d;
	q     = b + e(i-1);
	e(i)  = -f(i) * q;
end

% Trim the padded zero elements
e(1:2) = [];
f(1:2) = [];

% If we''ve converged, fill the rest of the elements with the limit
if Nend < N
	e((Nend+1):end) = e(Nend);
	f((Nend+1):end) = f(Nend);
end
% Done
