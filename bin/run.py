#!/usr/bin/env python3

import argparse
import logging
import pyinotify

from bb_live import FileEventHandler, SiteBuilder, PeriodicHiveAnalysis


def main():
    parser = argparse.ArgumentParser(description='Create a static beesbook live view website')
    parser.add_argument("source_dir", help="Path to directory where live preview images and " +
                                           "analysis files are stored", type=str)
    parser.add_argument("output_path", help="Path to static html output file", type=str)
    parser.add_argument("--interval", help="Ignore file events that occur more frequently than " +
                                           "this interval", default=30, type=int)
    parser.add_argument("--debug", help="Print debug logging messages", default=False, type=bool)

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO)

    wm = pyinotify.WatchManager()
    builder = SiteBuilder(args.output_path, min_interval=args.interval)
    handler = FileEventHandler(args.source_dir, builder, min_interval=args.interval)
    notifier = pyinotify.Notifier(wm, handler)
    wm.add_watch(args.source_dir, pyinotify.IN_CLOSE_WRITE)

    hive_analysis = PeriodicHiveAnalysis(interval=3600, builder=builder)
    hive_analysis.run_periodic()

    notifier.loop()

if __name__ == '__main__':
    main()