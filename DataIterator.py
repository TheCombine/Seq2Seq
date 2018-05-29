import numpy as np

class DataIterator(object):
    def __init__(self, IndicesObj, IndicesSrc):
        self.IndicesObj = IndicesObj
        self.IndicesSrc = IndicesSrc
        self.size = len(IndicesObj)
        self.pointer = 0;

    def GetNextBatch(self, n):
        n = min(n, len(self.IndicesObj) - self.pointer)
        if n <= 0:
            return None
        X, X_len = self._pad(self.IndicesObj, n)
        Y, Y_len = self._pad(self.IndicesSrc, n, appendStart=[1])
        Y_t, Y_len_t = self._pad(self.IndicesSrc, n, appendEnd=[2])
        self.pointer += n
        return [X, X_len, Y, Y_len, Y_t]

    def _pad(self, Indices, n, appendStart=None, appendEnd=None):
        appendLengths = 0
        if appendStart != None:
            appendLengths += len(appendStart)
        if appendEnd != None:
            appendLengths += len(appendEnd)

        line_lengths = [len(fn) + appendLengths for fn in Indices[self.pointer:self.pointer+n]]
        padded_lines = np.zeros([n, max(line_lengths)], np.int32)

        for i, padded_line in enumerate(padded_lines):
            offset = 0

            if appendStart != None:
                offset = len(appendStart)
                padded_line[:offset] = appendStart

            padded_line[offset:offset + len(Indices[self.pointer + i])] = Indices[self.pointer + i]
            offset += len(Indices[self.pointer + i])

            if appendEnd != None:
                padded_line[offset:offset + len(appendEnd)] = appendEnd

        return [padded_lines, line_lengths]