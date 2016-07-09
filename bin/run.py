#!/usr/bin/env python3

import argparse
import logging
import pyinotify

from bb_live import CreateLiveSiteHandler


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create a static beesbook live view website')
    parser.add_argument("source_dir", help="Path to directory where live preview images and " + \
                                           "analysis files are stored", type=str)
    parser.add_argument("output_file", help="Path to static html output file", type=str)
    parser.add_argument("interval", help="Ignore file events that occur more frequently than " + \
                                         "this interval", type=int)

    args = parser.parse_args()

    wm = pyinotify.WatchManager()
    handler = CreateLiveSiteHandler(args.output_file, args.source_dir, min_interval=args.interval)
    notifier = pyinotify.Notifier(wm, handler)
    wm.add_watch(args.source_dir, pyinotify.IN_CLOSE_WRITE)

    notifier.loop()
