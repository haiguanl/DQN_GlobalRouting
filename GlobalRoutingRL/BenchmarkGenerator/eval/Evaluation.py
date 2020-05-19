import os

if __name__ == "__main__":
    benchmark_num = 20
    for i in range(benchmark_num):
        command = "perl eval2008.pl test_benchmark_{num}.gr\
        test_benchmark_{num}.grAstar_solution".format(num=i+1)
        os.system(command)
