## SYDE 572 A4
Author: Cooper Ang (20768006)


Code was developed in  `python a4_ex1.py` for the CNN implementation (Exercise 1), but run on Google Colab using GPU
The notebook that was used in `572A4.ipynb`

I ran a total of 8 experiments, each with 5 epochs. The results are in the folder ex1_outputs (json files contain the raw results)

Note the file names inside the folder will be shorthand VGG#, where in the report I refer to them as VGG11 #

VGG11_1 - SGD, LR=0.1, Momentum 0.9, BS 128
VGG11_2 - SGD, LR=0.01, Momentum 0.9, BS 128
VGG11_3 - RMSprop, LR=0.1, Momentum 0.9, BS 128
VGG11_4 - Adam, LR=0.1, BS 128
VGG11_5 - RMSprop, LR=0.001, BS 128 (No Momentum)
VGG11_6 - Adam, LR=0.001, BS 128 
VGG11_7 - Sigmoid, SGD, LR=0.1, Momentum 0.9, BS 128
VGG11_8 - No dropout, SGD, LR=0.1, Momentum 0.9, BS 

MLP implemenation for Exercise 2 was run on CPU with  `python a4_ex2.py`
Exercise 2 results can be found in ex2_outputs

Please read comments in the code. Code is not explained in the report.