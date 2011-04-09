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
    Y(Nfft2+1) = Y(Nfft2+1) * lambda ./ (24 + lambda);
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
zs = sort(roots([1, lambda/6-4, 2/3*lambda+6, lambda/6-4, 1]));
elim = zs(1) + zs(2);
flim = zs(1) + zs(2);

denom = (1-flim) * (1 - elim*elim + 2 * flim + flim*flim);
glim  = flim * (1-flim) / denom;
hlim  = elim * flim / denom;
qlim  = (elim*elim*flim - flim*flim - flim*flim*flim)/denom;

trace = 6*Nw*glim - 8*(Nw - 1)*hlim + 2*(Nw - 2)*qlim;

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
    % have to be added in later.
    e = norm(Y(2:Nfft2) .* ...
                (1 - lambda * Hcos2(1:(end-1)) ./ ...
                 (Hcos1(1:(end-1)) + lambda * Hcos2(1:(end-1))))) * 2;
    e = e + norm(Y(1)) + norm(Y(Nfft2+1) * lambda ./ (24 + lambda));
else
    % When odd, the (Nfft-1)/2 middle elements, designated
    % 2:((Nfft-1)/2+1) are repeated and ONLY the 1st element needs
    % to be added in later.
    e = norm(Y(2:(Nfft2+1)) .* ...
             (1 - lambda * Hcos2 ./ ...
              (Hcos1 + lambda * Hcos2))) * 2;
    e = e + abs(Y(1)).^2;
end

gcv = e / trace;
end

end