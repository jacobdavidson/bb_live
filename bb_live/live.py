import logging
import os
import time
from base64 import b64encode
from collections import defaultdict
from datetime import datetime
from io import BytesIO

import matplotlib as mpl
import pandas as pd
import pyinotify
from jinja2 import Environment, PackageLoader
from numpy import rot90
from scipy.misc import imread, imsave

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from bb_binary import parse_image_fname_beesbook


def get_b64_uri(bytes, format):
    src = 'data:image/{0};base64,{1}'
    return src.format(format, b64encode(bytes.getvalue()).decode('utf-8'))

def get_image_bytes(image, format='jpeg'):
    b = BytesIO()
    imsave(b, image, format)
    return b

def get_fig_bytes(format='png', **kwargs):
    b = BytesIO()
    plt.savefig(b, bbox_inches='tight', format=format, **kwargs)
    return b


class CreateLiveSiteHandler(pyinotify.ProcessEvent):
    def __init__(self, output_fname, source_dir, template='index.html', min_interval=30,
                 analysis_metrics=('filename', 'smd', 'variance', 'noise'),
                 camera_rotations={0: 1, 1: -1, 2:1, 3:-1}):
        env = Environment(loader=PackageLoader('beesbook-live', 'templates'))
        self.template = env.get_template(template)
        self.min_interval = min_interval
        self.output_fname = output_fname

        self.access_history = defaultdict(lambda: datetime.fromtimestamp(0))
        self.uris = defaultdict(str)

        self.rotations = camera_rotations

        self.analysis_metrics = analysis_metrics
        self.analysis_paths = [os.path.join(source_dir, f) for f in
                               os.listdir(source_dir) if f.startswith('analysis')]

    def parse_analysis(self):
        dfs = []
        for fn in self.analysis_paths:
            analysis = pd.read_csv(fn, sep='\t', names=self.analysis_metrics)
            analysis['camIdx'] = [parse_image_fname_beesbook(s)[0] for s in analysis['filename']]
            analysis['datetime'] = [pd.datetime.fromtimestamp(
                    parse_image_fname_beesbook(s)[1]) for s in analysis['filename']]
            dfs.append(analysis)
        analysis = pd.concat(dfs)
        analysis.sort('datetime', inplace=True)
        return analysis

    def plot_analysis(self, analysis):
        for column in self.analysis_metrics[1:]:
            fig, ax = plt.subplots(1, figsize=(16, 4), facecolor='white')
            for camIdx in (0, 1, 2, 3):
                analysis[analysis.camIdx==camIdx].plot('datetime', column,
                                                       label='cam{}'.format(camIdx),
                                                       title=column, ax=ax)
            ax.legend(loc='upper left')
            ax.set_xlabel('time')
            self.uris[column] = get_b64_uri(get_fig_bytes(), format='png')
        plt.close('all')

    def process_image(self, path, fname):
        im = imread(path)
        camIdx = fname.split('.')[0][-1]
        im = rot90(im, self.rotations[int(camIdx)])
        self.uris['cam{}'.format(camIdx)] = get_b64_uri(get_image_bytes(im), format='jpeg')

    def process_IN_CLOSE_WRITE(self, event):
        path = os.path.join(event.path, event.name)
        now = datetime.now()
        last_access = self.access_history[path]
        time_since = now - last_access
        if time_since.seconds >= self.min_interval:
            update = True
            if event.name.startswith('cam'):
                logging.info('Updating {}'.format(event.name))
                self.access_history[path] = now
                time.sleep(1)
                self.process_image(path, event.name)
            elif event.name.startswith('analysis'):
                logging.info('Updating {}'.format(event.name))
                self.access_history[path] = now
                self.plot_analysis(self.parse_analysis())
            else:
                update = False
                logging.info('Event ignored: {}'.format(event))

            if update:
                html = self.template.render(last_updated=now, **self.uris)
                open(self.output_fname, 'w').write(html)
