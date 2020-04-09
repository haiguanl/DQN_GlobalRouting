import os

if __name__ == "__main__":
    benchmark_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,\
    27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,51,52,53,54,55,56,\
    57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,85,\
    86,87,88,89,91,92,93,94,95,96,97,98,99,100]
    for i in range(len(benchmark_num)):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.gr.DRLsolution".format(num=benchmark_num[i])
        os.system(command)
        command = "perl eval2008.pl test_benchmark_{num}.gr\
         test_benchmark_{num}.grAstar_solution".format(num=benchmark_num[i])
        os.system(command)




