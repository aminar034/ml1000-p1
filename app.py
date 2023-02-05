
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import sys
import argparse
import logging
from datetime import datetime
#
from flask import Flask, request, url_for, redirect, render_template, jsonify
#
from model import HotelReservationDataset
#
OPT_VERBOSE_HELP = "Display additional information about execution."
#
logger = logging.getLogger(__name__)
app = Flask(__name__)
#
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
    # Extract the data from the form
    logger.debug(request.form)
    # Number of adults
    no_of_adults = int(request.form['no_of_adults'])
    # Number of children
    no_of_children = int(request.form['no_of_children'])
    check_in_date_t = (
        int(request.form["check_in_year"]),
        int(request.form["check_in_month"]),
        int(request.form["check_in_day"]))
    check_out_date_t = (
        int(request.form["check_out_year"]),
        int(request.form["check_out_month"]),
        int(request.form["check_out_day"]))
    type_of_meal_plan = request.form['type_of_meal_plan']
    required_car_parking_space = request.form['required_car_parking_space']
    room_type_reserved = int(request.form['room_type_reserved'])

    check_in_dt = datetime(check_in_date_t[0], check_in_date_t[1], check_in_date_t[2])
    check_out_dt = datetime(check_out_date_t[0], check_out_date_t[1], check_out_date_t[2])

    check_in_s = check_in_dt.strftime("%Y-%m-%d")
    check_out_s = check_out_dt.strftime("%Y-%m-%d")

    # TODO: no_of_week_nights
    # TODO: no_of_weekend_nights
    # TODO: lead_time
    market_segment_type = request.form['market_segment_type']
    repeated_guest = int(request.form['repeated_guest'])
    no_of_previous_cancellations = int(request.form['no_of_previous_cancellations'])
    no_of_previous_bookings_not_canceled = int(request.form['no_of_previous_bookings_not_canceled'])
    avg_price_per_room = float(request.form['avg_price_per_room'])
    no_of_special_requests = int(request.form['no_of_special_requests'])
    booking_status = request.form['booking_status']


    logger.info(f"Received request with the following data items: ")
    logger.info(f"\tno_of_adults                    : {no_of_adults}")
    logger.info(f"\tno_of_children                  : {no_of_children}")
    logger.info(f"\tcheck_in_dt                     : {check_in_s}")
    logger.info(f"\tcheck_out_dt                    : {check_out_s}")
    logger.info(f"\trequired_car_parking_space      : {required_car_parking_space}")
    logger.info(f"\troom_type_reserved              : {room_type_reserved}")
    logger.info(f"\tmarket_segment_type             : {market_segment_type}")
    logger.info(f"\trepeated_guest                  : {repeated_guest}")
    logger.info(f"\tno_of_previous_cancellations    : {no_of_previous_cancellations}")
    logger.info(f"\tno_of_previous_bookings_not_canceled     : {no_of_previous_bookings_not_canceled}")
    logger.info(f"\tavg_price_per_room                       : {avg_price_per_room}")
    logger.info(f"\tno_of_special_requests                   : {no_of_special_requests}")
    logger.info(f"\tbooking_status                  : {booking_status}")

    return render_template("result.html")


def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="")
    parser.add_argument('-t', '--train',
                    dest="do_train",
                    action="store_true",
                    help="Train a new model without starting the web server.")
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

    # Train a new model
    if args.do_train:
        # Load the data
        datafile = os.path.abspath('./data/data.csv')
        dataset = HotelReservationDataset(_file=datafile, _target_col="booking_status")
        logger.info(f"{dataset.nb_rows} row(s) loaded from '{datafile}'.")
        # Generate the model
        logger.info("Selecting best classifier model...")
        model = dataset.best_model
        # Print information about the best model
        print(model)
        # Generate a file name based on the current date and time
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{now}"
        file_path = os.path.abspath(os.path.join(".", "models", file_name))
        # Save the best model to a file
        dataset.save_best_model_to(file_path)
        logger.info(f"Save model to '{file_path}.")
    # Otherwise, start the server
    else:
        # Load the latest model
        # TODO: Select the most recent model, or the model specified
        # TODO: Load the model
        # Start the server
        app.run(port=int(args.port), debug=bool(args.is_debug), threaded=True)

def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))

if __name__ == '__main__':
    entry_point()

