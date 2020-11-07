#ifndef HMM_HEADER
#define HMM_HEADER

#include <vector>
#include <stack>
#include <algorithm>

#include <iostream>

class HMM{
private:
  int N; // Number of hidden states
  int M; // Number of obervation symbols
  std::vector<long double> initial;
  std::vector<std::vector<long double>> transition;
  std::vector<std::vector<long double>> emission;

  void forward(std::vector<int>& O, std::vector<std::vector<long double>>& alpha);
  void backward(std::vector<int>& O, std::vector<std::vector<long double>>& beta);

public:
  HMM(int N);

  void setInitial(std::vector<long double>& initial);
  void setTransition(std::vector<std::vector<long double>>& transition);
  void setEmission(std::vector<std::vector<long double>>& emission);

  long double getLikelihood(std::vector<int>& O);

  std::vector<int> decode(std::vector<int>& O);

  void train(std::vector<int>& O);
};

HMM::HMM(int N, int M){
  this->N = N;
  this->M = M;
}
void HMM::setInitial(std::vector<long double>& initial){
  this->initial.clear();
  for(int j = 0; j < initial.size(); ++j)
    this->initial.push_back(initial[j]);
}
void HMM::setTransition(std::vector<std::vector<long double>>& transition){
  this->transition.clear();
  for(int j = 0; j < transition.size(); ++j){
    std::vector<long double> arr;
    for(int i = 0; i < transition[j].size(); ++i)
      arr.push_back(transition[j][i]);
    this->transition.push_back(arr);
  }
}
void HMM::setEmission(std::vector<std::vector<long double>>& emission){
  this->emission.clear();
  for(int j = 0; j < emission.size(); ++j){
    std::vector<long double> arr;
    for(int i = 0; i < emission[j].size(); ++i)
      arr.push_back(emission[j][i]);
    this->emission.push_back(arr);
  }
}

void HMM::forward(std::vector<int>& O, std::vector<std::vector<long double>>& alpha){
  int T = O.size();

  alpha.clear();
  alpha.resize(T);
  for(int t = 0; t < T; ++t) alpha[t].resize(N);

  for(int j = 0; j < N; ++j)
    alpha[0][j] = initial[j] * emission[j][O[0]];
  for(int t = 1; t < T; ++t){
    for(int j = 0; j < N; ++j){
      alpha[t][j] = 0;
      for(int i = 0; i < N; ++i)
        alpha[t][j] += alpha[t - 1][i] * transition[i][j] * emission[j][O[t]];
    }
  }
}
void HMM::backward(std::vector<int>& O, std::vector<std::vector<long double>>& beta){
  int T = O.size();

  beta.clear();
  beta.resize(T);
  for(int t = 0; t < T; ++t) beta[t].resize(N);

  for(int i = 0; i < N; ++i)
    beta[T - 1][i] = 1;
  for(int t = T - 2; t >= 0; --t){
    for(int i = 0; i < N; ++i){
      beta[t][i] = 0;
      for(int j = 0; j < N; ++j)
        beta[t][i] += transition[i][j] * emission[j][O[t + 1]] * beta[t + 1][j];
    }
  }
}

long double HMM::getLikelihood(std::vector<int>& O){
  int T = O.size();
  std::vector<std::vector<long double>> alpha;
  
  forward(O, alpha);

  long double p = 0;
  for(int i = 0; i < N; ++i)
    p += alpha[T - 1][i];
  return p;
}

std::vector<int> HMM::decode(std::vector<int>& O){
  int T = O.size();
  std::vector<int> estQ;
  
  std::vector<std::vector<long double>> v(T);
  std::vector<std::vector<int>> bt(T);
  for(int t = 0; t < T; ++t){
    v[t].resize(N);
    bt[t].resize(N);
  }

  for(int j = 0; j < N; ++j){
    v[0][j] = initial[j] * emission[j][O[0]];
    bt[0][j] = 0;
  }

  for(int t = 1; t < T; ++t){
    for(int j = 0; j < N; ++j){
      v[t][j] = v[t - 1][0] * transition[0][j] * emission[j][O[t]];
      bt[t][j] = 0;
      for(int i = 1; i < N; ++i)
        if(v[t][j] < v[t - 1][i] * transition[i][j] * emission[j][O[t]]){
          v[t][j] = v[t - 1][i] * transition[i][j] * emission[j][O[t]];
          bt[t][j] = i;
        }
    }
  }

  estQ.resize(T);
  estQ[T - 1] = 0;
  long double bestScore = v[T - 1][0];
  for(int i = 1; i < N; ++i)
    if(bestScore < v[T - 1][i]){
      estQ[T - 1] = i;
      bestScore = v[T - 1][i];
    }
  
  for(int t = T - 2; t >= 0; --t)
    estQ[t] = bt[t + 1][estQ[t + 1]];
  
  return estQ;
}

void HMM::train(std::vector<int>& O){
  int T = O.size();

  std::vector<std::vector<long double>> alpha, beta;
  forward(O, alpha);
  backward(O, beta);

  std::vector<std::vector<std::vector<long double>>> xi;
  xi.resize(T - 1);
  for(int t = 0; t < T - 1; ++t){
    xi[t].resize(N);
    for(int i = 0; i < N; ++i)
      xi[t][i].resize(N);
  }
  
  for(int t = 0; t < T - 1; ++t){
    long double OProb = 0;
    for(int j = 0; j < N; ++j)
      OProb += alpha[t][j] * beta[t][j];
    
    for(int i = 0; i < N; ++i)
      for(int j = 0; j < N; ++j){
        xi[t][i][j] = alpha[t][i] * transition[i][j] * emission[j][O[t + 1]] * beta[t + 1][j];
        xi[t][i][j] /= OProb;
      }
  }

  std::vector<std::vector<long double>> a;
  a.resize(N);
  for(int i = 0; i < N; ++i)
    a[i].resize(N);
  
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      long double numerator = 0;
      long double denominator = 0;

      for(int t = 0; t < T - 1; ++t)
        numerator += xi[t][i][j];
      
      for(int t = 0; t < T - 1; ++t)
        for(int k = 0; k < N; ++k)
          denominator += xi[t][i][k];
      
      a[i][j] = numerator / denominator;
    }
  }

  std::vector<std::vector<long double>> gamma;
  gamma.resize(T);
  for(int t = 0; t < T; ++t)
    gamma[t].resize(N);
  
  for(int t = 0; t < T; ++t){
    long double OProb = 0;
    for(int j = 0; j < N; ++j)
      OProb += alpha[t][j] * beta[t][j];
    
    for(int j = 0; j < N; ++j)
      gamma[t][j] = alpha[t][j] * beta[t][j] / OProb;
  }

  std::vector<std::vector<long double>> b;
  b.resize(N);
  for(int j = 0; j < N; ++j)
    b[j].resize(M);
  
  for(int j = 0; j < N; ++j){
    long double totProb = 0;
    for(int t = 0; t < T; ++t)
      totProb += gamma[t][j];
    
    for(int t = 0; t < T; ++t)
      b[j][O[t]] += gamma[t][j];

    for(int i = 0; i < M; ++i)
      b[j][i] /= totProb;
  }

  transition = a;
  emission = b;
}

#endif