""" This file contains gobal configurations.

    @author: jhuthmacher
"""

from datetime import datetime
import logging as log
from pathlib import Path
path = 'logs/'
folder = Path(path)
folder.mkdir(parents=True, exist_ok=True)

# Custom logger configuration.
logFormatter = log.Formatter('%(asctime)s %(levelname)s: %(message)s',
                             datefmt='%d.%m.%Y %H:%M:%S')

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%d.%m.%Y %H:%M:%S',
                level=log.INFO,
                # filename=f'{path}/{datetime.now().date()}.log'
                )

fh = log.FileHandler(f'{path}/{datetime.now().date()}.log')
fh.setFormatter(logFormatter)
log.getLogger().addHandler(fh)
