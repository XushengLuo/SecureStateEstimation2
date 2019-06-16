#!/bin/bash

#for r#  in 0.15 0.12 0.1 0.07 0.04 0.02
# do
#    echo "----------------------r = ${r}------------------------"
#    for ((n=0;n<20;n++))
#    do
#        /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Documents/GitHub/RRGLTL/OptPlan4MulR.py ${r}
#    done
# done
for ((n=0;n<100000;n++))
do
     echo "-- --------------------n = ${n}------------------------"
    /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Documents/GitHub/SecureStateEstimation/sse_without_print.py
done
