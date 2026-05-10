functions {
  vector alr(vector x, int i) {
    int l = num_elements(x);
    vector[l - 1] out;
    int pos = 1;

    for (j in 1:l) {
      if (j != i) {
        out[pos] = log(x[j]) - log(x[i]);
        pos += 1;
      }
    }
    return out;
  }

  vector alrinv(vector x, int i) {
    int l = num_elements(x) + 1;
    vector[l] logits;
    int pos = 1;

    for (j in 1:l) {
      if (j == i) {
        logits[j] = 0;
      } else {
        logits[j] = x[pos];
        pos += 1;
      }
    }
    return softmax(logits);
  }
}

data {
  int<lower=2> T;
  int<lower=2> C;
  array[T] simplex[C] Y;
  int<lower=1, upper=C> ref;
  int<lower=0> P;
  int<lower=0> Q;
}

transformed data {
  int M = max(P, Q);
  array[T] vector[C - 1] alr_Y;

  for (t in 1:T) {
    alr_Y[t] = alr(Y[t], ref);
  }
}

parameters {
  array[P] matrix[C - 1, C - 1] A;
  array[Q] matrix[C - 1, C - 1] B;
  real phi_intercept;
}

transformed parameters {
  array[T] vector[C - 1] eta;
  array[T] vector[C] alpha;

  if (M > 0) {
    for (t in 1:M) {
      eta[t] = alr_Y[t];
      alpha[t] = exp(phi_intercept) * alrinv(eta[t], ref);
    }
  }

  if (T > M) {
    for (t in (M + 1):T) {
      vector[C - 1] ar = rep_vector(0, C - 1);
      vector[C - 1] ma = rep_vector(0, C - 1);

      for (p in 1:P) {
        ar += A[p] * alr_Y[t - p];
      }

      for (q in 1:Q) {
        ma += B[q] * (alr_Y[t - q] - eta[t - q]);
      }

      eta[t] = ar + ma;
      alpha[t] = exp(phi_intercept) * alrinv(eta[t], ref);
    }
  }
}

model {
  for (p in 1:P) {
    to_vector(A[p]) ~ normal(0, 1);
  }

  for (q in 1:Q) {
    to_vector(B[q]) ~ normal(0, 1);
  }

  phi_intercept ~ normal(7, 1.5);

  if (T > M) {
    for (t in (M + 1):T) {
      Y[t] ~ dirichlet(alpha[t]);
    }
  }
}

generated quantities {
  vector[T - M] log_lik;

  if (T > M) {
    for (t in (M + 1):T) {
      log_lik[t - M] = dirichlet_lpdf(Y[t] | alpha[t]);
    }
  }
}
