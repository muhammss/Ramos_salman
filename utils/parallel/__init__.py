from itertools import chain
from multiprocessing import Process, Queue
from operator import itemgetter
import os

__all__ = ['parmap']

def parmap(target, varying, constant=(), reduction=None, ncpus=None, unwrap=True):
	"""parallel map
	- 'target' a function to be called on all inputs
	- 'varying' a list of tuples of arguments to pass to the target function
	- 'constant' a tuple of arguments to pass to the target function
	-'reduction' optional function to reduce the output
	-'ncpus': optionally, number of parallel workers to use
	-'unwrap': if false, treat 'varying' as a list of single arguments, rather than as a 
	list of argument tuples
	The target functions must be written so that the variable arguments come
	before the constant ones.
	"""
	if not cpus:
		ncpus = os.cpu_count()

	if not unwrap:
		varying = [(v, ) for v in varying]

	chunks = split (varying, cpus)
	q = Queue()

	#Start  a process for each worker, and run the delegate function
	process = []
	for pid, chunk in enumerate(chunks):
		p = Process (
			target=delegate,
			args = (pid, q, target, chunk, constant, reduction),
		)
		p.start()
		processes.append((pid, p))

	#Grab the result from each fucntion and shut them down
	result = [q.get() for _ in range(ncpus)]

	for pid, p in processes:
		p.join()
	#Sort result by pid

	result = [v for _, v in sorted(result, key=itemgetter(0))]

	#return either reduction or flattened list
	if reduction:
		return reduction(result)
	return list(chain.from_iterable(result))