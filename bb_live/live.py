import logging
import pickle
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
import numpy as np
import pytz
from numpy import rot90
from scipy.misc import imread, imsave, imresize
from datetime import datetime
from scipy.interpolate import interp1d
from skimage import exposure

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from bb_binary import parse_image_fname_beesbook, get_timezone
from pipeline import Pipeline
from pipeline.objects import Image, FinalResultOverlay, IDs, LocalizerInputImage, CrownOverlay, SaliencyOverlay
from pipeline.stages import ResultCrownVisualizer


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

def localize(dt):
    return pytz.timezone('UTC').localize(dt).astimezone(get_timezone())

def get_localtime():
    return localize(datetime.now())

class ImgTemplate:
    def __init__(self, id=None, src=None, alt=None, header=None, reload_interval=30000):
        self.id = id
        self.src = src
        self.alt = alt
        self.header = header
        self.reload_interval = reload_interval


class RowTemplate:
    def __init__(self, num_cols):
        self.images = [ImgTemplate() for _ in range(num_cols)]


class TabTemplate:
    def __init__(self, name, num_rows, num_cols, active_on_load=False):
        self.name = name
        self.rows = [RowTemplate(num_cols) for _ in range(num_rows)]
        self.set_active_on_load(active_on_load)


    def set_active_on_load(self, active_flag=False):
        self.div_class_extras = 'in active' if active_flag else ''
        self.li_class_extras = 'active' if active_flag else ''


class SectionTemplate:
    def __init__(self, name, header=None, tabs=None, grid_class='col-xs-12 col-sm-12 col-md-6 col-lg-6'):
        self.name = name
        self.header = header
        self.grid_class = grid_class
        if tabs is not None:
            self.tabs = tabs
        else:
            self.tabs = []


class SiteBuilder:
    def __init__(self, output_path, template='index.html', min_interval=30):
        env = Environment(loader=PackageLoader('bb_live', 'templates'), trim_blocks=True)
        self.template = env.get_template(template)
        self.min_interval = min_interval
        self.output_path = output_path
        self.output_fname = os.path.join(output_path, template)
        self.uris = defaultdict(str)

        image_tabs = [TabTemplate(name, 2, 2) for name in ('detections', 'decodings', 'inputs', 'saliencies')]
        image_tabs[0].set_active_on_load(True)
        image_section = SectionTemplate('images', header=None, tabs=image_tabs)

        hive_tabs = [TabTemplate('bees', 1, 1, active_on_load=True)]
        hive_section = SectionTemplate('hive', header='Hive statistics over time',
                                       tabs=hive_tabs, grid_class='col-xs-12 col-sm-12 col-md-12 col-lg-12')

        metrics_tabs = [TabTemplate(name, 1, 1) for name in ('smd', 'variance', 'noise', 'contrast', \
                                                             'cratio', 'cratioMinMax')]
        metrics_tabs[0].set_active_on_load(True)
        metrics_section = SectionTemplate('metrics', header='Image statistics over time',
                                          tabs=metrics_tabs, grid_class='col-xs-12 col-sm-12 col-md-12 col-lg-12')

        self.sections = [image_section, hive_section, metrics_section]

    def get_image_template(self, section_name, tab_name, idx):
        section = [s for s in self.sections if s.name == section_name]
        assert(len(section) == 1)
        section = section[0]
        tab = [t for t in section.tabs if t.name == tab_name]
        assert(len(tab) == 1)
        tab = tab[0]
        row_idx, col_idx = divmod(int(idx), len(tab.rows))
        return tab.rows[row_idx].images[col_idx]

    def update_uri(self, key, value):
        self.uris[key] = value

    def save_image(self, name, im, format):
        fname = name + '.' + format
        imsave(os.path.join(self.output_path, fname), im)

    def save_figure(self, name, format):
        fname = name + '.' + format
        plt.savefig(os.path.join(self.output_path, fname),
                    bbox_inches='tight', format=format)

    def save_tab_image(self, name, tab, section, idx, format, header=None):
        id = '{}_{}'.format(tab, name)
        fname = '{}.{}'.format(id, format)
        self.update_uri(name, '{}?{}'.format(fname, get_localtime().timestamp()))
        image_template = self.get_image_template(section, tab, idx)
        image_template.id = id
        image_template.src = fname
        image_template.alt = '{} live preview'.format(id)
        image_template.header = header

    def build(self):
        html = self.template.render(last_updated=get_localtime(), sections=self.sections, **self.uris)
        open(self.output_fname, 'w').write(html)


class ImageHandler:
    def __init__(self, site_builder, pipeline_config, camera_rotations={0: 1, 1: -1, 2:1, 3:-1}):
        self.builder = site_builder
        self.rotations = camera_rotations
        self.pipeline = Pipeline([Image], [LocalizerInputImage, FinalResultOverlay, CrownOverlay,
                                           IDs, SaliencyOverlay],
                                 **pipeline_config)
        self.crown = ResultCrownVisualizer()

        self.detections_path = 'detections.pkl'
        if os.path.isfile(self.detections_path):
            self.detections = pickle.load(open(self.detections_path, 'rb'))
        else:
            self.detections = []

    def run_pipeline(self, image):
        results = self.pipeline([image])
        return results

    def plot_detections(self, num_plot_samples=1000, std_window_size=25):
        detections = pd.DataFrame(self.detections, columns=('datetime', 'camIdx', 'id'))

        minTs = detections.datetime.min().timestamp()
        maxTs = detections.datetime.max().timestamp()
        fig, ax = plt.subplots(1, figsize=(16, 4), facecolor='white')

        sumValues = None

        for camIdx in (0, 1, 2, 3):
            x = np.array([ts.timestamp() for ts in \
                          detections[detections.camIdx == str(camIdx)].groupby('datetime').id.agg(len).keys()])
            y = detections[detections.camIdx == str(camIdx)].groupby('datetime').id.agg(len).values
            if len(x) < 2:
                continue
            f = interp1d(x, y, fill_value='extrapolate')
            dts = np.arange(minTs, maxTs, step=((maxTs - minTs) / num_plot_samples)).astype(np.float64)
            values = f(dts)
            df = pd.DataFrame([(localize(datetime.fromtimestamp(dt)), val) for dt, val in zip(dts, values)],
                              columns=('datetime', 'cam{}'.format(camIdx)))
            df.plot(ax=ax)

            if sumValues is None:
                sumValues = values
            else:
                sumValues += values

        if sumValues is not None:
            df = pd.DataFrame([(datetime.fromtimestamp(dt), val) for dt, val in zip(dts, sumValues)],
                              columns=('datetime', 'combined'))
            df.plot(ax=ax, yerr=df.combined.rolling(window=std_window_size, min_periods=0).std(),
                    ecolor=tuple(list(sns.color_palette()[4]) + [0.3]))

        ax.legend(loc='upper left')
        ax.set_xlabel('time')

        ax.locator_params(nbins=12)

        ax.set_xticklabels([dt.strftime('%a %H:%M:%S') for dt in df.datetime[ax.get_xticks()[:-1]]])
        ax.set_title('Number of tagged bees in colony')

        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)

        self.builder.update_uri('detections', get_b64_uri(get_fig_bytes(), format='png'))
        self.builder.save_figure('bees_detections', format='png')
        self.builder.save_tab_image('detections', 'bees', 'hive', idx=0, format='png')
        plt.close('all')

    def process_image(self, path, fname):
        im = imread(path)
        camIdx = fname.split('.')[0][-1]
        im = rot90(im, self.rotations[int(camIdx)])
        orig_im = np.copy(im)
        img_adapteq = exposure.equalize_adapthist(im, clip_limit=0.01)
        im = np.round(img_adapteq * 255).astype(np.uint8)

        results = self.run_pipeline(im)
        dt = get_localtime()
        for id in results[IDs]:
            self.detections.append((dt, camIdx, int(''.join([str(c) for c in np.round(id).astype(np.int)]))))
        pickle.dump(self.detections, open(self.detections_path, 'wb'))
        self.plot_detections()

        img_with_overlay = self.crown.add_overlay(img_adapteq, results[CrownOverlay])
        saliency_overlay = results[SaliencyOverlay]

        self.builder.save_image('detections_cam{}'.format(camIdx), results[FinalResultOverlay], format='jpeg')
        self.builder.save_image('decodings_cam{}'.format(camIdx), img_with_overlay, format='jpeg')
        self.builder.save_image('saliencies_cam{}'.format(camIdx), saliency_overlay, format='jpeg')
        self.builder.save_image('inputs_cam{}'.format(camIdx), orig_im, format='jpeg')

        self.builder.save_tab_image('cam{}'.format(camIdx), 'detections', 'images',
                                    camIdx, format='jpeg', header='Cam{}'.format(camIdx))
        self.builder.save_tab_image('cam{}'.format(camIdx), 'decodings', 'images',
                                    camIdx, format='jpeg', header='Cam{}'.format(camIdx))
        self.builder.save_tab_image('cam{}'.format(camIdx), 'saliencies', 'images',
                                    camIdx, format='jpeg', header='Cam{}'.format(camIdx))
        self.builder.save_tab_image('cam{}'.format(camIdx), 'inputs', 'images',
                                    camIdx, format='jpeg', header='Cam{}'.format(camIdx))


class AnalysisHandler:
    def __init__(self, source_dir, site_builder,
                 analysis_metrics=('filename', 'smd', 'variance', 'noise', 'contrast', 'cratio', 'cratioMinMax')):
        self.builder = site_builder
        self.analysis_metrics = analysis_metrics
        self.analysis_paths = [os.path.join(source_dir, f) for f in
                               os.listdir(source_dir) if f.startswith('analysis')]

    def parse_analysis(self):
        dfs = []
        for fn in self.analysis_paths:
            analysis = pd.read_csv(fn, sep='\t', names=self.analysis_metrics)
            analysis['camIdx'] = [parse_image_fname_beesbook(s)[0] for s in analysis['filename']]
            analysis['datetime'] = [localize(pd.datetime.fromtimestamp(
                    parse_image_fname_beesbook(s)[1])) for s in analysis['filename']]
            dfs.append(analysis)
        analysis = pd.concat(dfs)
        analysis.sort_values('datetime', inplace=True)
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
            self.builder.update_uri(column, get_b64_uri(get_fig_bytes(), format='png'))
            self.builder.save_figure('{}_metrics'.format(column), format='png')
            self.builder.save_tab_image('metrics', column, 'metrics', idx=0, format='png')
        plt.close('all')

    def update(self):
        self.plot_analysis(self.parse_analysis())


class FileEventHandler(pyinotify.ProcessEvent):
    def __init__(self, source_dir, site_builder, pipeline_config, min_interval):
        self.access_history = defaultdict(lambda: get_timezone().localize(datetime.fromtimestamp(0)))
        self.builder = site_builder
        self.hndl_analysis = AnalysisHandler(source_dir, self.builder)
        self.hndl_image = ImageHandler(self.builder, pipeline_config)
        self.min_interval = min_interval

    def process_IN_CLOSE_WRITE(self, event):
        path = os.path.join(event.path, event.name)
        now = get_localtime()
        last_access = self.access_history[path]
        time_since = now - last_access
        if time_since.seconds >= self.min_interval:
            update = True
            self.access_history[path] = now
            if event.name.startswith('cam'):
                logging.info('Updating {}'.format(event.name))
                time.sleep(1)
                try:
                    self.hndl_image.process_image(path, event.name)
                except Exception as err:
                    logging.error(err)
            elif event.name.startswith('analysis'):
                logging.info('Updating {}'.format(event.name))
                try:
                    self.hndl_analysis.update()
                except Exception as err:
                    logging.error(err)
            else:
                logging.info('Event ignored: {}'.format(event))
                update = False
                del self.access_history[path]

            if update:
                self.builder.build()
