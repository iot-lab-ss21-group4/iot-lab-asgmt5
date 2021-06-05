import argparse
import sys

import data_generator
import lr_model
import lstm_model
import sarima_model

DATAGEN_SUBCOMMAND = "datagen"
LINREG_SUBCOMMAND = "linreg"
SARIMAX_SUBCOMMAND = "sarimax"
LSTM_SUBCOMMAND = "lstm"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="Subcommands")
    datagen_parser = subparser.add_parser(DATAGEN_SUBCOMMAND, help="Subcommand to generate artificial training data.")
    datagen_parser.set_defaults(subcommand_func=data_generator.add_arguments)
    datagen_parser.set_defaults(subcommand_kwargs={"parser": datagen_parser})

    linreg_parser = subparser.add_parser(LINREG_SUBCOMMAND, help="Subcommand to use linear regression model.")
    linreg_parser.set_defaults(subcommand_func=lr_model.add_arguments)
    linreg_parser.set_defaults(subcommand_kwargs={"parser": linreg_parser})

    sarimax_parser = subparser.add_parser(SARIMAX_SUBCOMMAND, help="Subcommand to use SARIMAX model.")
    sarimax_parser.set_defaults(subcommand_func=sarima_model.add_arguments)
    sarimax_parser.set_defaults(subcommand_kwargs={"parser": sarimax_parser})

    lstm_parser = subparser.add_parser(LSTM_SUBCOMMAND, help="Subcommand to use LSTM model.")
    lstm_parser.set_defaults(subcommand_func=lstm_model.add_arguments)
    lstm_parser.set_defaults(subcommand_kwargs={"parser": lstm_parser})

    sub_args = parser.parse_args(sys.argv[1:2])
    sub_args.subcommand_func(**sub_args.subcommand_kwargs)
    args = parser.parse_args()
    args.func(args)
