# -*- coding: utf-8 -*-
from collections import namedtuple
from os import path, makedirs
import pickle
import uuid

Index = namedtuple('Index',['datapath', 'table', 'block_size', 'num_files', 'len'])
FileData = namedtuple('FileData', ['file', 'start', 'end'])

class DataStoreError(Exception):
    pass

class DataStore:
    """ A container class for storing and loading big data sequences.

    Data sequences are treated as lists. Data are persisted on files with
    extension .store, items are accessed by loading one file at time.

    """

    @classmethod
    def load(self, fn):
        """Loads datastore from file."""

        try:
            with open(fn, 'rb') as f:
                index = pickle.load(f)

            datapath = index.datapath
            table = index.table
            block_size = index.block_size
            num_files = index.num_files
            len = index.len

        except:
            raise DataStoreError('Impossible to load data.')

        name = fn.split('.')[0]
        data_store = DataStore(name, datapath=datapath, block_size=block_size)
        data_store.table = table
        data_store._num_files = num_files
        data_store._len = len

        return data_store

    def __init__(self, name, datapath=None, block_size=1000000):

        self.name = name
        self._datapath = datapath
        self._block_size = block_size
        self._data = []
        self._offset = 0
        self.table = []
        self._num_files = 0
        self._len = 0
        self._iter_count = 0

    def add(self, item):
        """Adds an object to container. """

        self._data.append(item)
        self._len += 1

        if len(self._data)>=self._block_size:
            self._persist()

    def commit(self):
        """Persists pending data, creates index and saves it to file. """

        if self._data:
            self._persist()

        index = Index(self._datapath, self.table, self._block_size, self._num_files, self._len)

        try:
            # pickle data to file
            fn = f'{self.name}.store'

            if self._datapath:
                fn = path.abspath(path.join(self._datapath, fn))

            makedirs(path.dirname(fn), exist_ok=True)
            with open(fn, 'wb') as f:
                pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

        except:
            raise DataStoreError(f'{fn}: Impossible to save data index.')

    def __add__(self, items):
        """Concatenates with list of tuple. """
        if isinstance(items, list):
            self._data += items
        elif isinstance(items, tuple):
            self._data += list(items)
        else:
            raise DataStoreError('Can only concatenate list or tuple.')

        self._len += len(items)

        if len(self._data)>=self._block_size:
            self._persist()

        return self

    def _persist(self):
        """Persists block data to disk. """

        fn = uuid.uuid4().hex

        if self._datapath:
            fn = path.abspath(path.join(self._datapath, fn))

        try:
            # pickle data to file
            makedirs(path.dirname(fn), exist_ok=True)
            with open(fn, 'wb') as f:
                pickle.dump(self._data, f, pickle.HIGHEST_PROTOCOL)
        except:
            raise DataStoreError(f'{fn}: Impossible to save data block.')

        else:

            data_size = len(self._data)

            filedata = FileData(fn, self._offset, self._offset+data_size-1)
            self.table.append(filedata)

            # update counters
            self._num_files+=1
            self._offset += data_size
            self._data = []

        return

    def __getitem__(self, key):
        """Magic function to access data element by key. """

        if isinstance(key, slice):
            inds = list(range((key.start or 0), (key.stop or len(self)), (key.step or 1)))
        else:
            inds = [key]

        # files in table follow an ascendent order for indexes
        items = []
        ii = 0
        ind = inds[ii]

        found = False
        for filedata in self.table:

            fn = filedata.file
            fn_inds = []

            while filedata.start<=ind and filedata.end>=ind:

                fn_inds.append(ind-filedata.start)

                if ii<len(inds)-1:
                    ii += 1
                    ind = inds[ii]
                else:
                    break

            if fn_inds:
                if not found: found = True
                items += self._load_from_file(fn, fn_inds)

        if len(items)==1: items = items[0]

        if not found:
            raise DataStoreError('Item not found.')

        return items

    def _load_from_file(self, fn, inds):
        """Loads data from file. """
        try:
            with open(fn, 'rb') as f:
                data = pickle.load(f)
        except:
            raise DataStoreError(f'Impossible to load data block {fn}.')

        return [data[ind] for ind in inds]

    def __len__(self):
        return self._len

    def __iter__(self):
        self._iter_count = 0
        return self

    def __next__(self):
        if self._iter_count <= self._len:
            item = self.__getitem__(self._iter_count)
            self._iter_count += 1
        else:
            raise StopIteration
        return item
