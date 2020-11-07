#include <iostream>
#include <vector>
#include "HMM/hmm.h"
using namespace std;

typedef long double ldb;

ostream& operator<<(ostream& os, const vector<int>& dt){
  for(auto t: dt)
    os << t << " ";
  return os;
}

int main(){
  HMM hmm(2, 2);
  vector<ldb> init{
    0.5, 0.5
  };
  vector<vector<ldb>> A{
    {0.9, 0.1},
    {0.9, 0.1}
  };
  vector<vector<ldb>> B{
    {0.9, 0.1},
    {0.1, 0.9}
  };

  hmm.setInitial(init);
  hmm.setTransition(A);
  hmm.setEmission(B);

  vector<int> O1{0, 0, 0};
  vector<int> O2{1, 1, 1};
  vector<int> O3{0, 1, 1, 1, 1, 1};
  cout << "Before any train: " << endl;
  cout << hmm.getLikelihood(O1) << endl;
  cout << hmm.getLikelihood(O2) << endl;
  cout << hmm.getLikelihood(O3) << endl;

  cout << hmm.decode(O1) << endl;
  cout << hmm.decode(O2) << endl;
  cout << hmm.decode(O3) << endl;

  for(int j = 0; j < 10; ++j){
    cout << "training HMM for O3 " << j << endl;
    hmm.train(O3);
    cout << hmm.getLikelihood(O1) << endl;
    cout << hmm.getLikelihood(O2) << endl;
    cout << hmm.getLikelihood(O3) << endl;

    cout << hmm.decode(O1) << endl;
    cout << hmm.decode(O2) << endl;
    cout << hmm.decode(O3) << endl;
  }
  return 0;
}