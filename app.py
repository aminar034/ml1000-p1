
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import sys
import argparse
import logging
#
from flask import Flask, request, url_for, redirect, render_template, jsonify
#
OPT_VERBOSE_HELP = "Display additional information about execution."

logger = logging.getLogger(__name__)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="")
    parser.add_argument('-p', '--port',
                    dest="port",
                    default=5000,
                    help="Port to listen on for HTTP requests.")
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action="store_true",
                        dest="is_debug",
                        help=OPT_VERBOSE_HELP)

    logging.basicConfig(level=logging.DEBUG)

    # Parse command-line arguents
    args = parser.parse_args(args=argv[1:])
    print("")

    # Start the server
    app.run(port=int(args.port), debug=bool(args.is_debug), threaded=True)

def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))

if __name__ == '__main__':
    entry_point()

