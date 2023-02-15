#pragma once

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

template <typename TYPE>
void read_ts(char *file_name, TYPE *data, unsigned int n)
{
    
    FILE * file = fopen(file_name, "rt");
    assert(file != NULL);
    
    int i = 0;
    
    while (!feof(file))
    {
        //fscanf(file, "%lf\n", &data[i]);
        fscanf(file, "%f\n", &data[i]);
        i++;
    }
    
    fclose(file);
    
    for (int j = i; j < n; j++)
        data[j] = FLT_MAX;
}


template <typename TYPE>
void write_range_discord_profile(char *outfile_name, TYPE *profile, unsigned int N, unsigned int m)
{
    // open for output in append mode (create a new file only if the file does not exist)
    ofstream outfile(outfile_name, ios::app);
    
    outfile << m << ";";
    
    // Send data to the stream
    for (int i = 0; i < N; i++)
    {
        if (i != N-1)
            outfile << profile[i] << ";";
        else
            outfile << profile[i] << "\n";
    }
    
    outfile.close();
}


template <typename TYPE>
void write_discords(char *outfile_name, vector<pair<TYPE, int>> top_k_discords, unsigned int K, unsigned int m)
{
    // open for output in append mode (create a new file only if the file does not exist)
    ofstream outfile(outfile_name, ios::app);
    
    // Send data to the stream
    for (int i = 0; i < K; i++)
    {
            outfile << m << ";" << top_k_discords[i].second << ";" << top_k_discords[i].first << "\n";
    }
    
    outfile.close();
}
