import os

if __name__ == "__main__":
    benchmark_num = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    for i in range(len(benchmark_num)):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.gr.DRLsolution".format(num=benchmark_num[i])
        os.system(command)
        command = "perl eval2008.pl test_benchmark_{num}.gr\
         test_benchmark_{num}.grAstar_solution".format(num=benchmark_num[i])
        os.system(command)




