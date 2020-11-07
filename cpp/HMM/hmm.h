#ifndef HMM_HEADER
#define HMM_HEADER

#include <vector>
#include <stack>
#include <algorithm>

#include <iostream>

class HMM{
private:
  int N; // Number of hidden states
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

HMM::HMM(int N){
  this->N = N;
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
  
}

#endif