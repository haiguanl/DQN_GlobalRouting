import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser('Benchmark Generator Parser')
    parser.add_argument('--benchNumber',type=int,\
        dest='benchmarkNumber',default=20)
    parser.add_argument('--gridSize',type=int,dest='gridSize',default=16)
    parser.add_argument('--netNum',type=int,dest='netNum',default=5)
    parser.add_argument('--capacity',type=int,dest='cap',default=4)
    parser.add_argument('--maxPinNum',type=int,dest='maxPinNum',default=5)
    parser.add_argument('--reducedCapNum',type=int,dest='reducedCapNum',default=1)

    return parser.parse_args()


if __name__ == '__main__':
	# Remember to copy results to other directory when running new parameters

	filename = None
	args = parse_arguments()
	benNum = args.benchmarkNumber
	gridSize = args.gridSize; netNum = args.netNum
	cap = args.cap; maxPinNum = args.maxPinNum
	reducedCapNum = args.reducedCapNum

	# Generating problems module (A*)
	# Make sure previous benchmark files: "benchmark","capacityplot",
	# and 'solution' are removed
	os.system('rm -r benchmark')
	os.system('rm -r capacityplot_A*')
	os.system('rm -r solutionsA*')
	os.system('rm -r solutionsDRL')
	os.chdir('BenchmarkGenerator')
	# os.chdir('..')
	print('**************')
	print('Problem Generating Module')
	print('**************')
	os.system('python BenchmarkGenerator.py --benchNumber {benNum} --gridSize {gridSize}\
	 --netNum {netNum} --capacity {cap} --maxPinNum {maxPinNum} --reducedCapNum {reducedCapNum}'\
	 .format(benNum=benNum,gridSize=gridSize,\
	 	netNum=netNum,cap=cap,maxPinNum=maxPinNum,reducedCapNum=reducedCapNum))

	# Solve problems with DRL 
	os.chdir('..') # Go back to main folder
	os.system('python Router.py')
	# !!! Code the results for heatmap with DRL solution

	# Evaluation of DRL and A* solution


	# Plot results 
	# WL and OF with sorted A* results; difference in WL

	

