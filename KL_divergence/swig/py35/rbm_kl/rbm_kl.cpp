#include<iostream>
#include<fstream>
#include "math.h"
#include <map>
#include"stdlib.h"
#include"rbm_kl.h"
using namespace std;


double calculate_H_data(string filename)
{
	typedef map<int, int> MAP_INT_INT;
	MAP_INT_INT mc_sample_map;
	MAP_INT_INT::const_iterator iElementFound;

	unsigned int num_of_sample_i, num_sample = 0;
	int sample_i;

	double H_data = 0;
	string sample_i_str;

	ifstream infile((const char*)filename.c_str());
	
	while (!infile.eof())
	{
		
		getline(infile, sample_i_str);
		num_sample++;
		sample_i = atoi(sample_i_str.c_str());
		iElementFound = mc_sample_map.find(sample_i);
		if (iElementFound != mc_sample_map.end())
		{
			num_of_sample_i = iElementFound->second;
			mc_sample_map[sample_i] = (num_of_sample_i + 1);
		}
		else
			mc_sample_map.insert(pair<int, int>(sample_i, 1));


	}
	cout << "please make sure your csv have " << num_sample << " samples"<<endl;



	MAP_INT_INT::const_iterator sum_delta_s1_s2;

	for (sum_delta_s1_s2 = mc_sample_map.begin(); sum_delta_s1_s2 != mc_sample_map.end(); ++sum_delta_s1_s2)
	{

		H_data += (sum_delta_s1_s2->second)* log((sum_delta_s1_s2->second) / (1.0*num_sample));

	}
	return -1*H_data / num_sample;

}

