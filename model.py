#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import logging

import pandas as pd
from pycaret.classification import *
#
logger = logging.getLogger(__name__)
#


class HotelReservationDataset(object):
    """
    | Column        |   Description         |
    |---------------|-----------------------|
    | `Booking_ID`       | unique identifier of each booking |
    | `no_of_adults`     | Number of adults                  |
    | `no_of_children`   | Number of Children                |
    | `no_of_weekend_nights` | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
    | `no_of_week_nights`    | Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
    | `type_of_meal_plan`    |` Type of meal plan booked by the customer:
    | `required_car_parking_space` | Does the customer require a car parking space? (0 - No, 1- Yes)
    | `room_type_reserved` | Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
    | `lead_time`          | Number of days between the date of booking and the arrival date
    | `arrival_year`       | Year of arrival date
    | `arrival_month`      | Month of arrival date
    | `arrival_date`       | Date of the month
    | `market_segment_type`| Market segment designation.
    | `repeated_guest`     |` Is the customer a repeated guest? (0 - No, 1- Yes)
    | `no_of_previous_cancellations`         | Number of previous bookings that were canceled by the customer prior to the current booking
    | `no_of_previous_bookings_not_canceled` | Number of previous bookings not canceled by the customer prior to the current booking
    | `avg_price_per_room`     | Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
    | `no_of_special_requests` | Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
    | `booking_status`         | Flag indicating if the booking was canceled or not.
    """

    BOOK_STATUS_MAPPING = {'Not_Canceled': 0, 'Canceled': 1}
    MEAL_PLAN_MAPPING = {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2}

    def __init__(self, _file:str, _target_col:str, _training_size=0.8, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file)
        self._df.reset_index()

        # Remove unneeded columns
        if len(_drop_cols) > 0:
            logger.info(f"Removing {len(_drop_cols)} column(s) from the original data set.")
            self._df = self._df.drop(_drop_cols)

        # Create mappings to integers
        if "booking_status" in self._df:
            self._df['booking_status'] = self._df['booking_status'].map(HotelReservationDataset.BOOK_STATUS_MAPPING)
            #self._df['booking_status'] = self._df['booking_status'].astype(int)
        if "type_of_meal_plan" in self._df:
            self._df['type_of_meal_plan'] = self._df['type_of_meal_plan'].map(HotelReservationDataset.MEAL_PLAN_MAPPING)
            #self._df['type_of_meal_plan'] = self._df['type_of_meal_plan'].astype(int)

        # Do some cleaning
        #null_rows = self._df.isnull().sum()
        #if null_rows > 0:
        #    logger.info(f"{null_rows} empty row(s) will be removed.")
        #    self._df = self._df.dropna()

        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        self._clf=setup(data=self._df, target=_target_col)

        # https://pycaret.gitbook.io/docs/get-started/functions/train#compare_models
        self._best_model = compare_models()

    @property
    def filename(self) -> str:
        return self._datafile

    @property
    def classifier(self):
        return self._clf

    @property
    def dataframe(self):
        return self._df

    @property
    def nb_rows(self) -> int:
        return self.dataframe.shape[0]

    @property
    def nb_cols(self) -> int:
        return self.dataframe.shape[1]

    @property
    def best_model(self):
        return self._best_model

    def save_best_model_to(self, _filename:str) -> None:
        # Save the best model to a file
        save_model(self.best_model, _filename)