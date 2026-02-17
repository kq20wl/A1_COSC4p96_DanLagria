# A1_COSC4p96_DanLagria
Assignment 1 of class COSC 4P96

All runs are in 10_DanLagria_COSC4p69
10 is for version
the previous are bugged or hard to work with
Indent may be weird, due to use of google collab since it was used to run it in parallel

In the file
- Find the file path for cifar in stage 1 (second code block) if there is trouble with getting the files  
- At the very top, change the weight size:
    - 10% for all related data
        - comment out train method Z_score and uncomment Min_max for those results
          Note: Minmax is not used at or during stage 2 f, restore to original to work as intended
    - That's it, learning rate and and others are in their own isolated test to get a feel 
      or resuluts and optimize before seed testing
    

Requirements:
- Tensor flow
- Numpy

Or just run it in google collab, which works as well, just remeber to upload the cifar data
also available in github @ https://github.com/kq20wl/A1_COSC4p96_DanLagria