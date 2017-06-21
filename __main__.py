import click
from importlib import import_module
import logging
import numpy as np
from tqdm import tqdm
from vtk import vtkProbeFilter

from ramos import io
from ramos.reduction import Reduction
from ramos.utils.vtk import write_to_file
from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import mv_dot, vv_dot

@click.group()
@click.option('--verbosity', '-v', type=click.Choice(['debug', 'info', 'warning', 'error',
	'critical']), default = 'info')

def main(verbosity):
		logging.basicConfig (
			format = '{asctime} {levelname: <10} {message}',
			datefmt = '%H:%M',
			style = '{',
			level = verbosity.upper(),
			)
@main.command()
@click.argument ('data', type = io.DataSourceType())
def summary(data):
	print(data)

@main.command
@click.option ('--field', '-f', 'fields', type=str, multiple=True, help='Fields to read')
@click.option('--error', '-e', type=float, default=0.05, help='Relative error threshold to achieve')
@click.option('--out', '-o', type=str, default='out', help='Name of output')
@click.option('--min-modes', type=int, default=10, help='Minimum number of modes to write')
@click.argument('sources', type=io.DataSourceType(), nargs=-1)

def reduce(fields, error, out, min_modes, sources):
	"""Calculate a reduced basis"""
	sink = sources[0].sink(out)
	r = Reduction (sources, fields. sink, out, min_modes, error)
	r.reduce()

@main.command()