%module rbm_kl  
%include "std_string.i"
%inline %{  
#include "rbm_kl.h" 

%}  
double calculate_H_data(std::string filename); 