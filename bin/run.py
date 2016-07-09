#!/usr/bin/env python3

import logging
from io import BytesIO
import os
from base64 import b64encode
from collections import defaultdict
from datetime import datetime
import numpy as np
from numpy import rot90
import time
import pandas as pd
import pytz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from jinja2 import Environment, PackageLoader
from scipy.misc import imread, imsave, imrotate
import pyinotify

_TIMEZONE = pytz.timezone('Europe/Berlin')

def get_timezone():
    return _TIMEZONE

def to_timestamp(dt):
    try:
        return dt.timestamp()
    except AttributeError:  # python 2
        utc_naive = dt.replace(tzinfo=None) - dt.utcoffset()
        timestamp = (utc_naive - datetime(1970, 1, 1)).total_seconds()
    return timestamp

def parse_image_fname_beesbook(fname):
    basename = os.path.basename(fname)
    _, camIdxStr, datetimeStr, usStr = basename.split('_')

    camIdx = int(camIdxStr)
    year = int(datetimeStr[:4])
    month = int(datetimeStr[4:6])
    day = int(datetimeStr[6:8])

    hour = int(datetimeStr[8:10])
    minute = int(datetimeStr[10:12])
    second = int(datetimeStr[12:14])
    us = int(usStr)

    dt = datetime(year, month, day, hour, minute, second, us)
    ts = to_timestamp(get_timezone().localize(dt))
    return camIdx, ts


class CreateLiveSiteHandler(pyinotify.ProcessEvent):
    def __init__(self, output_fname, template='index.html', min_interval=30):
        env = Environment(loader=PackageLoader('beesbook-live', 'templates'))
        self.template = env.get_template(template)
        self.min_interval = min_interval
        self.output_fname = output_fname

        self.access_history = defaultdict(lambda: datetime.fromtimestamp(0))
        self.uris = defaultdict(str)

        self.rotations = {0: 1, 1: -1, 2:1, 3:-1}

        self.analysis_paths = (
                '/mnt/storage/hauke/live/analysis01.txt',
                '/mnt/storage/hauke/live/analysis23.txt')

    def b64image(self, image):
        b = BytesIO()
        imsave(b, image, 'jpeg')
        src = 'data:image/jpeg;base64,{0}'
        return src.format(b64encode(b.getvalue()).decode('utf-8'))

    def parse_analysis(self):
        dfs = []
        for fn in self.analysis_paths:
            analysis = pd.read_csv(fn, sep='\t', names=['filename', 'smd', 'variance', 'contrast', 'noise'])
            analysis['camIdx'] = [parse_image_fname_beesbook(s)[0] for s in analysis['filename']]
            analysis['datetime'] = [pd.datetime.fromtimestamp(
                    parse_image_fname_beesbook(s)[1]) for s in analysis['filename']]
            dfs.append(analysis)
        analysis = pd.concat(dfs)
        analysis.sort('datetime', inplace=True)

        for column in ('smd', 'variance', 'contrast', 'noise'):
            fig, ax = plt.subplots(1, figsize=(16, 4), facecolor='white')
            for camIdx in (0, 1, 2, 3):
                analysis[analysis.camIdx==camIdx].plot('datetime', column, label='cam{}'.format(camIdx),
                                                       title=column, ax=ax)
            ax.legend(loc='upper left')
            ax.set_xlabel('Time')
            b = BytesIO()
            plt.savefig(b, bbox_inches='tight', format='png')
            src = 'data:image/png;base64,{0}'
            self.uris[column] = src.format(b64encode(b.getvalue()).decode('utf-8'))
        plt.close('all')

    def process_IN_CLOSE_WRITE(self, event):
        path = os.path.join(event.path, event.name)
        now = datetime.now()
        last_access = self.access_history[path]
        time_since = now - last_access
        if time_since.seconds >= self.min_interval:
            update = True
            if event.name.startswith('cam'):
                camIdx = event.name.split('.')[0][-1]
                logging.info('Updating cam {}'.format(camIdx))
                self.access_history[path] = now
                time.sleep(1)
                im = imread(path)
                im = rot90(im, self.rotations[int(camIdx)])
                self.uris['cam{}'.format(camIdx)] = self.b64image(im)
            elif event.name.startswith('analysis'):
                logging.info('Updating {}'.format(event.name))
                self.parse_analysis()
            else:
                update = False
                logging.info('Event ignored: {}'.format(event))

            if update:
                html = self.template.render(last_updated=now, **self.uris)
                open(self.output_fname, 'w').write(html)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s', level=logging.INFO)

    wm = pyinotify.WatchManager()
    handler = CreateLiveSiteHandler('/home/ben/biorobotics-web-live/index.html')
    notifier = pyinotify.Notifier(wm, handler)
    wm.add_watch('/mnt/storage/hauke/live/', pyinotify.IN_CLOSE_WRITE)

    notifier.loop()
