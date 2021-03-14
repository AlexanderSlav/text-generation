import argparse
import json
from loguru import logger
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i_p', '--input_data_path', type=str, help='input path to json dataset file', required=True)
    parser.add_argument('-o_p', '--output_data_path', type=str, help='output path to txt dataset file', required=True)
    args = parser.parse_args()
    return args


def main(args):
    logger.info('Starting convertion')
    with open(args.input_data_path, "r") as input_file:
        input_data = json.load(input_file)
    with open(args.output_data_path, "w") as output_file:
        for article in tqdm(input_data):
            output_file.write(f'{article["text"]}\n\n')
    logger.info('Finish convertion')


if __name__ == "__main__":
    args = parse_args()
    main(args)