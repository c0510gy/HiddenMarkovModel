# HiddenMarkovModel
HMM (Hidden Markov Model) that supports various operations such as Likelihood, Decoding, Learning, etc.

The algorithm was implemented according to [this](https://c0510gy.github.io/2020/11/04/Hidden-Markov-Model.html)

## How to use

The following example demonstrates initializing basic parameters.

```cpp
int numberOfHiddenStates = 2;
int numberOfObservableSymbols = 2;

HMM hmm(numberOfHiddenStates, numberOfObservableSymbols);

// Initial probability distribution
vector<ldb> init{
  0.5, 0.5
};
// Transition probability matrix
vector<vector<ldb>> A{
  {0.9, 0.1},
  {0.9, 0.1}
};
// Observation likelihoods
vector<vector<ldb>> B{
  {0.9, 0.1},
  {0.1, 0.9}
};

hmm.setInitial(init);
hmm.setTransition(A);
hmm.setEmission(B);
```

You can get the likelihood of an observation sequence as follows.

```cpp
vector<int> O1{0, 0, 0};
vector<int> O2{1, 1, 1};
vector<int> O3{0, 1, 1, 1, 1, 1};
cout << hmm.getLikelihood(O1) << endl;
cout << hmm.getLikelihood(O2) << endl;
cout << hmm.getLikelihood(O3) << endl;
```

Also, you can run the decode operation.

```cpp
vector<int> estStates = hmm.decode(O3);
```

The following example shows how to get updated parameters after the training.

```cpp
hmm.train(O3);
vector<vector<long double>> transition = hmm.getTransition();
vector<vector<long double>> emission = hmm.getEmission();
```
