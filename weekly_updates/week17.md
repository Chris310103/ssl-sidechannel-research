# Weekly Update

## Done
Completely run the on-the-fly experiment using 1,000 training traces, U=300, 100 epochs, and CNN_Best_Norm. 

Results:

- All 256 TripletPower models and kNN classifiers were successfully trained and evaluated.

- The accuracy-based distinguisher failed to recover the correct key (0x15). At 5,000 test traces, the correct key ranked 84/256 with 22.70% accuracy.

- However, the classifier trained under the correct key hypothesis successfully achieved Profiling Key Rank 0 using the same 5,000 test traces.

## Ongoing
Checking the code of foundation model sca

## To Do
Read the paper of "Early Exit"
