import os

if __name__ == "__main__":
    benchmark_num = [3,4,6,7,8,9,11,12,15,17,19,20,\
    21,22,23,24,25,31,33,34,35,40,42,43,45,49,50,51,\
    54,56,57,58,59,60,61,62,63,64,66,70,71,72,74,75,\
    81,82,83,84,87,88,89,90,91,92,95,]

    for i in range(len(benchmark_num)):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.gr.DRLsolution".format(num=benchmark_num[i])
        os.system(command)
        command = "perl eval2008.pl test_benchmark_{num}.gr\
         test_benchmark_{num}.grAstar_solution".format(num=benchmark_num[i])
        os.system(command)




