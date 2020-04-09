import os

if __name__ == "__main__":
    benchmark_num = [3,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,
    25,28,29,31,33,34,35,36,37,38,40,42,43,45,46,47,49,50,51,52,54,56,57,58,
    59,60,61,62,63,64,66,67,68,70,71,72,74,75,81,82,83,84,87,88,89,90,91,92,
    95,96]
    for i in range(len(benchmark_num)):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.gr.DRLsolution".format(num=benchmark_num[i])
        os.system(command)
        command = "perl eval2008.pl test_benchmark_{num}.gr\
         test_benchmark_{num}.grAstar_solution".format(num=benchmark_num[i])
        os.system(command)




