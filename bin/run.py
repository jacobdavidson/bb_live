#!/usr/bin/env python3

import argparse
import json
import logging
import pyinotify

from bb_live import FileEventHandler, SiteBuilder


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create a static beesbook live view website')
    parser.add_argument("source_dir", help="Path to directory where live preview images and " + \
                                           "analysis files are stored", type=str)
    parser.add_argument("output_path", help="Path to static html output file", type=str)
    parser.add_argument("pipeline_config", help="Path to json config file for the pipeline", type=str)
    parser.add_argument("--interval", help="Ignore file events that occur more frequently than " + \
                                         "this interval", default=30, type=int)

    args = parser.parse_args()

    pipeline_config = json.loads(open(args.pipeline_config, 'r').read())

    wm = pyinotify.WatchManager()
    builder = SiteBuilder(args.output_path, min_interval=args.interval)
    handler = FileEventHandler(args.source_dir, builder, pipeline_config, min_interval=args.interval)
    notifier = pyinotify.Notifier(wm, handler)
    wm.add_watch(args.source_dir, pyinotify.IN_CLOSE_WRITE)

    notifier.loop()
