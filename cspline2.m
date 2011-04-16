function [x, lambda] = cspline2(y)

Ny = length(y);
Nw = Ny - 2;


%% Compute the fft
Nfft = Ny;
Neven = mod(Nfft, 2) == 0;
Y = fft(y, Nfft)';


%% Compute the omegas for this type of fft
if Neven
    Nfft2 = Nfft/2;
else
    Nfft2 = (Nfft-1)/2;
end
w = (1:Nfft2)/Nfft2 * pi;
Hcos2 = 2 + cos(w);
Hcos1 = 12*(1 - cos(w)).^2;

%% Minimize the GCV via lambda
opts   = optimset('Display', 'off');
sigma  = fminbnd(@iterate, 0, 1);
lambda = sigma^2/(1-sigma^2);

%% Compute the final Y
if Neven
    Y(2:Nfft2) = Y(2:Nfft2) * lambda .* Hcos2(1:(end-1)) ./ ...
        (Hcos1(1:(end-1)) + lambda * Hcos2(1:(end-1)));
    Y((Nfft2+2):end) = conj(Y(Nfft2:-1:2));
    Y(Nfft2+1) = Y(Nfft2+1) * lambda ./ (48 + lambda);
else
    Y(2:(Nfft2+1)) = Y(2:(Nfft2+1)) * lambda .* Hcos2 ./ ...
        (Hcos1 + lambda * Hcos2);
    Y((Nfft2+2):end) = conj(Y((Nfft2+1):-1:2));
end

x = ifft(conj(Y));


function gcv = iterate(sigma)

%% Decompress lambda
lambda = sigma^2/(1-sigma^2);


%% Compute the steady-state trace
zs = sort(roots([1, lambda/6-4, 2/3*lambda+6, lambda/6-4, 1]), 'ascend');
elim = zs(1) + zs(2);
flim = zs(1)*zs(2);

denom = (1-flim) * ((1+flim)^2 - elim^2);
glim = flim * (1+flim) / denom;
hlim = elim * flim / denom;
qlim = elim * hlim - flim * glim;

trace = 6*glim - 8*hlim + 2*qlim;


%% Compute the error energy Y * (1 - H)
%  (This takes advantage of Parseval's theorem for the DFT which
%  states that n * ||y-x||^2 == (Y-X)(Y-X)^* which is quite
%  convenient for computing the gcv. Since we're dealing with real
%  signals we can abuse symmetry properties in the DFT, however,
%  it's important to note that these vary slightly depending on
%  whether Nfft is even or odd.

if Neven
    % When even, the (Nfft/2-1) middle elements, designated
    % 2:(Nfft/2) are repeated. The 1st and (Nfft/2+1)th elements
    % have to be added in later, although the 1st is multiplied by
    % 0 in the (1-H) term.
    e = sum(abs(Y(2:Nfft2) .* (1 - lambda * Hcos2(1:(end-1)) ./ ...
                               (Hcos1(1:(end-1)) + lambda * ...
                                Hcos2(1:(end-1))))).^2);
    e = e + abs(Y(Nfft2+1) * (1 - lambda/(48 + lambda)))^2;
else
    % When odd, the (Nfft-1)/2 middle elements, designated
    % 2:((Nfft-1)/2+1) are repeated and ONLY the 1st element needs
    % to be added in later.
    e = sum(abs(Y(2:(Nfft2+1)) .* ...
                (1 - lambda * Hcos2 ./ ...
                 (Hcos1 + lambda * Hcos2))).^2) * 2;
    e = e;
end

gcv = e/trace^2;
end

end