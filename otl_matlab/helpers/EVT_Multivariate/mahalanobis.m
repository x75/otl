function d = mahalanobis(X, Mu, C)
%MAHALANOBIS Mahalanobis distance.
%   D = MAHALANOBIS(X, MU, C) returns the Mahalanobis distance between
%   the length p vectors X and MU given the p by p covariance matrix
%   C. If omitted, it is assumed that C is the identity matrix
%   EYE(p). If either X or MU is an n by p matrix, D will be returned
%   as an n by g matrix where n is the number of rows in X and g is
%   the number of rows in MU where each entry i, j corresponds to the
%   mahalanobis distance between row i of X and row j of MU. If MU is
%   simply 0, it is treated as the origin from which Mahalanobis
%   distance to X is calculated. C must be a positive, definite,
%   symmetric matrix.
%
%   The Mahalanobis distance between vectors X(i,:) and MU(j,:) is
%   defined as:
%
%   D(i,j) = ((X(i,:) - MU(j,:))'*INV(C)*(X(i,:) - MU(j,:))).^(1/2)

%   Copyright (c) 1999 Michael Kiefte.

%   $Log$

error(nargchk(2, 3, nargin))

if isempty(X) | ~isa(X, 'double') | ~isreal(X) | ...
      any(any(isnan(X) | isinf(X)))
  error(['X must be a vector or matrix of real, finite numeric' ...
	 ' doubles.'])
elseif length(X) == prod(size(X))
  X = X(:)';
elseif ndims(X) ~= 2
  error('If X is a matrix, it must be a 2-d array.')
end

[n p] = size(X);

if isempty(Mu) | ~isa(Mu, 'double') | ~isreal(Mu) | ...
      any(any(isnan(Mu) | isinf(Mu)))
  error(['Mu must be a vector or matrix of real, finite numeric' ...
	 ' doubles.'])
elseif length(Mu) == prod(size(Mu))
  Mu = Mu(:)';
elseif ndims(Mu) ~= 2
  error('If MU is a matrix, it must be a 2-d array.')
end

if length(Mu) == 1 & Mu == 0
  Mu = zeros(1, p);
elseif size(Mu, 2) ~= p
  error('Number of features in MU and X must be the same.')
end

g = size(Mu, 1);

if ~isempty(C)
  if ~isa(C, 'double') | ~isreal(C) | ndims(C) ~= 2 | ...
	any(any(isnan(C) | isinf(C))) 
    error('C must be a matrix of real, finite numeric doubles.')
  elseif any(size(C) ~= p)
    error(['Number of rows and columns in C must match number of' ...
	   ' features in X and MU.'])
  end

  try
    S = inv(chol(C));
  catch
    error(['Covariance matrix C must be positive, definite,' ...
	   ' symmetric.'])
  end

  Xs = X*S;
  Ms = Mu*S;
else
  Xs = X;
  Ms = Mu;
end

d = sqrt(repmat(sum(Xs.^2, 2), 1, g) - 2*Xs*Ms' + ...
	 repmat(sum(Ms'.^2), n, 1));
