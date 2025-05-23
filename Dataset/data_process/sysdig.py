import os
import errno
import subprocess

import pandas as pd

SYSDIG = 'sysdig'
READ = '-r'
UTF = 'utf-8'
EVENT_LENGTH = 20
# NOISY_SCAP_METADATA = ['container', 'json', 'expecting', 'falcosecurity-libs']
NOISY_SCAP_METADATA = []
COLUMNS = ['timestamp', 'cpuid', 'syscall']
# COLUMNS = ['timestamp', 'syscall', 'args']

class NoSyscallsFound(Exception):
    def __init__(self, scap, message="The scap has no syscalls"):
        self.scap = scap
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.scap} -> {self.message}'
    pass


class CorruptedFile(Exception):
    def __init__(self, scap, message="The scap file is either corrupted or no data has been written"):
        self.scap = scap
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.scap} -> {self.message}'
    pass


class Sysdig:
    def __init__(self):
        pass

    def _sysdig_stream(self, scap):
        if os.path.exists(scap):
            process = subprocess.Popen([SYSDIG, READ, str(scap)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            while True:
                raw_evt = process.stdout.readline()
                yield raw_evt

                if process.poll() is not None:
                    break

            os.system('stty sane')

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), scap)

    def _sysdig_output(self, raw_evt):
        evt_info = raw_evt.decode(UTF).split()
        timestamp = str(evt_info[1][:-3])
        cpuid = str(evt_info[2])
        syscall = evt_info[6]
        # args = list(evt_info[7:])
        # evt = [timestamp, syscall, args]
        evt = [timestamp, cpuid, syscall]
        return evt

    def process_scap(self, scap):
        sysdig_evts = []

        if os.path.getsize(scap) > 0:

            for raw_evt in self._sysdig_stream(scap):

                noise = [x for x in NOISY_SCAP_METADATA if x in str(raw_evt)]

                if not noise and len(str(raw_evt)) > EVENT_LENGTH:
                    sysdig_evts.append(self._sysdig_output(raw_evt))

            scap_df = pd.DataFrame(sysdig_evts, columns=COLUMNS)

            if scap_df.empty:
                raise NoSyscallsFound(scap)
            else:
                return scap_df

        else:
            raise CorruptedFile(scap)