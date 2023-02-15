#pragma once

#define DATA_TYPE float

// constants for preprocessing and pruning off the subsequences
#define BLOCK_SIZE 1024

// constants for the 1st phase Candidate Selection and the 2nd phase Discord Refinement
#ifndef SEGMENT_N
#define SEGMENT_N 512
#endif
