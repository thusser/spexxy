import glob
import logging
import multiprocessing
import os
import pandas as pd

from .main import MainRoutine, FilesRoutine
from .object import create_object
from .utils.log import setup_log, shutdown_log


class Application(object):
    def __init__(self, config, filenames=None, ncpus=None, output=None, resume=False):
        # store it
        self._config = config
        self._filenames = filenames
        self._ncpus = ncpus
        self._output = output
        self._resume = resume

    def run(self):
        """Run application."""

        # init objects
        log = logging.getLogger('spexxy.main')
        log.info('Creating all objects...')
        objects = self._create_objects(log=log)
        log.info('Finished creating objects.')

        # initialize main routine
        log.info('Initializing main routine...')
        main = create_object(self._config['main'], objects=objects, log=log)

        # what type is main?
        if isinstance(main, FilesRoutine):
            # run on files
            self._run_on_files(log, main)
        elif isinstance(main, MainRoutine):
            # just run it
            main()

    def _run_on_files(self, log: logging.Logger, main: FilesRoutine):
        """Run the given FilesRoutine

        Args:
            log: Logger to use
            main: Routine to run
        """

        # filenames?
        if self._filenames is None:
            log.info('Nothing to do, going to bed...')

        # expand list of spectra
        filenames = []
        for f in self._filenames:
            if '*' in f or '?' in f:
                filenames.extend(glob.glob(f))
            else:
                filenames.append(f)

        # get columns
        columns = main.columns()

        # output csv?
        if self._output is not None:
            if self._resume and os.path.exists(self._output):
                # loading pre-existing data
                log.info('Loading results from existing output file...')
                data = pd.read_csv(self._output, index_col=False)

                # do columns match?
                if columns != list(data.columns.values)[1:]:
                    log.error('Columns in existing output file do not match request, please delete it.')
                    raise RuntimeError

                # filter files
                log.info('Filtering finished files...')
                filenames = list(filter(lambda filename: filename not in data['Filename'].values, filenames))

            else:
                # write new header
                log.info('Writing new output file...')
                with open(self._output, 'w') as f:
                    f.write('Filename,' + ','.join(columns) + '\n')

        # sort and count files
        self._filenames = sorted(filenames)
        self._total = len(self._filenames)
        log.info('Found a total of %d files to process.', len(self._filenames))

        # anything to do?
        if self._total == 0:
            log.info('Nothing to do, going to bed...')
            return

        # init
        pool = None
        if self._ncpus is None:
            # no, run sequentially
            log.info("Running analysis sequentially.")

        else:
            # number of cpus
            nprocs = min(self._ncpus, self._total)

            # yes, create pool of workers
            log.info("Starting analysis in parallel on %d CPUs..." % nprocs)
            pool = multiprocessing.Pool(nprocs)

        # loop
        for i, filename in enumerate(self._filenames, 1):
            # parallel?
            if self._ncpus is None:
                self._run_single(i, filename)
            else:
                pool.apply_async(self._run_single, (i, filename), error_callback=self.error)

        # join pool
        if pool is not None:
            pool.close()
            pool.join()

        # finished
        log.info('Finished.')

    def error(self, exception):
        logging.getLogger('spexxy.main').error('Something went wrong.', exc_info=exception)

    def _run_single(self, idx, filename):
        # main logger
        main_log = logging.getLogger('spexxy.main')

        # log
        main_log.info('(%i/%i) Starting on file %s...', idx, self._total, filename)

        # file exists?
        if not os.path.exists(filename):
            main_log.error('(%i/%i) File does not exist.', idx, self._total, filename)
            return

        # init file logger, show stdout output only if we're running on a single cpu
        log = setup_log('spexxy.fit', filename.replace('.fits', '.log'), stream=(self._ncpus is None))

        # create objects
        log.info('Creating all objects...')
        objects = self._create_objects(log=log)
        log.info('Finished creating objects.')

        # initialize main routine
        log.info('Initializing main routine...')
        main: FilesRoutine = create_object(self._config['main'], objects=objects, log=log)

        # init components
        log.info('Setting initial values...')
        if 'components' not in objects or objects['components'] is None:
            objects['components'] = {}
        for name, cmp in objects['components'].items():
            cmp.init(filename)

        # start fit
        results = None
        try:
            # do the fit
            log.info('Starting fit...')
            results = main(filename)
        except:
            log.exception('Exception during execution of fit.')

        # write result
        if self._output is not None and results is not None:
            with open(self._output, 'a') as f:
                # write filename
                f.write('%s' % filename)

                # write results
                if len(results) > 0:
                    f.write(',' + ','.join(['' if r is None else str(r) for r in results]))

                # write line break
                f.write('\n')

        # shutdown logger
        main_log.info('(%i/%i) Finished file %s...', idx, self._total, filename)
        log.info('Finished fit.')
        shutdown_log('spexxy.fit')

    def _create_objects(self, log=None):
        # create objects
        objects = {}
        for group, value in self._config.items():
            # don't create "main"
            if group != 'main':
                # all other groups are nested
                objects[group] = {}
                for name, config in value.items():
                    objects[group][name] = create_object(config, objects=objects, name=name, log=log)

        # finished
        return objects
