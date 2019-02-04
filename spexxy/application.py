import glob
import logging
import multiprocessing
import os
import pandas as pd

from .object import spexxyObject
from .utils.log import setup_log, shutdown_log


class Application(object):
    def __init__(self, config, filenames, ncpus=None, output=None, resume=False):
        # store it
        self._config = config
        self._filenames = filenames
        self._ncpus = ncpus
        self._output = output
        self._resume = resume

        # expand list of spectra
        self._filenames = []
        for f in filenames:
            if '*' in f or '?' in f:
                self._filenames.extend(glob.glob(f))
            else:
                self._filenames.append(f)

        # init objects
        log = logging.getLogger('spexxy.main')
        log.info('Creating all objects...')
        objects = self._create_objects(log=log)
        log.info('Finished creating objects.')

        # initialize main routine
        log.info('Initializing main routine...')
        main = spexxyObject.create_object(self._config['main'], objects=objects, log=log)

        # get columns
        columns = main.columns()

        # output csv?
        if output is not None:
            if resume and os.path.exists(output):
                # loading pre-existing data
                log.info('Loading results from existing output file...')
                data = pd.read_csv(output, index_col=False)

                # do columns match?
                if columns != list(data.columns.values)[1:]:
                    log.error('Columns in existing output file do not match request, please delete it.')
                    raise RuntimeError

                # filter files
                log.info('Filtering finished files...')
                self._filenames = list(filter(lambda filename: filename not in data['FILENAME'].values,
                                              self._filenames))

            else:
                # write new header
                log.info('Writing new output file...')
                with open(output, 'w') as f:
                    f.write('Filename,' + ','.join(columns) + '\n')

        # sort and count files
        self._filenames = sorted(self._filenames)
        self._total = len(self._filenames)
        log.info('Found a total of %d files to process.', len(self._filenames))

    def run(self):
        # get logger
        log = logging.getLogger('spexxy.main')

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
        main = spexxyObject.create_object(self._config['main'], objects=objects, log=log)

        # init components
        log.info('Setting initial values...')
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
                    objects[group][name] = spexxyObject.create_object(config, objects=objects, name=name, log=log)

        # finished
        return objects
