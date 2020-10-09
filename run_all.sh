#!/bin/bash

#for r#  in 0.15 0.12 0.1 0.07 0.04 0.02
# do
#    echo "----------------------r = ${r}------------------------"
#    for ((n=0;n<20;n++))
#    do
#        /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Documents/GitHub/RRGLTL/OptPlan4MulR.py ${r}
#    done
# done
#for ((n=0;n<100000;n++))
#do
#     echo "-- --------------------n = ${n}------------------------"
#    /usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Documents/GitHub/SecureStateEstimation/sse_without_print.py
#done

for p in 100
do
    echo "-----------------------p = ${p}-----------------------------"

    rm -R ./data
    mkdir ./data
    rm -R /Users/chrislaw/Box\ Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/Random\ examples/Test2_sensors
    mkdir /Users/chrislaw/Box\ Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/Random\ examples/Test2_sensors
    /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/generate_large_case.py ${p}

    # /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/noise_bound.py ${p}


    for noise in 0 1
    do
        for selection in 0 1
        do
            for percent in 1 2 3
            do
                /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/sse.py ${p} ${noise} ${selection} ${percent}
            done
        done
    done

    # for noise in 0
    # do
    #     for selection in 1
    #     do
    #         for percent in 3
    #         do
#                 /Applications/MATLAB_R2017b.app/bin/matlab -nodesktop -nosplash -r "cd('/Users/chrislaw/Box Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/Random examples'); scalability_test_from_python(${p},${noise},${selection},${percent});exit"
    #         done
    #     done
    # done

    # for noise in 0 1
    # do
    #     for selection in 0 1
    #     do
    #         for percent in 1 2 3
    #         do
    #             /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/MIQP.py ${p} ${noise} ${selection} ${percent}
    #         done
    #     done
    # done

done
# for p in 20
# do
#     echo "-----------------------p = ${p}-----------------------------"
#     for n in 200
#     do
#         echo "-----------------------n = ${n}-----------------------------"

#         rm -R ./data
#         mkdir ./data
#         rm -R /Users/chrislaw/Box\ Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/Random\ examples/Test2_sensors
#         mkdir /Users/chrislaw/Box\ Sync/Research/SSE_Automatica2019/Imhotep-smt-master/ImhotepSMT/Examples/Random\ examples/Test2_sensors
#         /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/generate_case_for_complexity.py ${p} ${n}
#         for noise in 1
#         do
#             for selection in 0
#             do
#                 for percent in 3
#                 do
#                     /usr/local/Cellar/python/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/chrislaw/Github/SecureStateEstimation2/sse_complexity.py ${p} ${noise} ${selection} ${percent} ${n}
#                 done
#             done
#         done
#     done
# done
