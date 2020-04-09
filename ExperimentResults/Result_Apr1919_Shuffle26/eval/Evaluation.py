import os

if __name__ == "__main__":
    benchmark_num = [3,4,5,6,7,10,11,12,13,15,20,21,22,23,28,31,36,38,39,40,42,43,44,48,50,\
    51,52,55,60,61,62,64,65,66,67,69,71,72,73,75,78,79,85,86,87,88,91,93,94,96,97,98,100]
    for i in range(len(benchmark_num)):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.gr.DRLsolution".format(num=benchmark_num[i])
        os.system(command)
        command = "perl eval2008.pl test_benchmark_{num}.gr\
         test_benchmark_{num}.grAstar_solution".format(num=benchmark_num[i])
        os.system(command)




