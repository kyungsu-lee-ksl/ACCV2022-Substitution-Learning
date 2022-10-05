import argparse

def getArgParser():

    parser = argparse.ArgumentParser(description='Overview of the system')

    parser.add_argument("-b", "--batch", "--batch_size", dest="batch", required=False, type=int, default=1, help="batch size to train models")
    parser.add_argument("-lr", "--learning_rate", dest="lr", required=False, default=1e-3, help='learning rate to train models')
    parser.add_argument("-e", "--epoch", dest='epoch', required=False, type=int, default=100, help="Number of epochs to train models")
    parser.add_argument('-w', "--width", dest='w', required=False, default=512, help='width of input images', type=int)
    parser.add_argument("--height", dest='h', required=False, default=512, help='height of input images', type=int)
    parser.add_argument("-c", '--channel', dest='channel', required=False, default=1, help='channels of input images', type=int)
    parser.add_argument('-i', '--interval', dest='save_interval', required=False, default=5, type=int, help='Number of epochs to save weights of models')
    parser.add_argument('-s', '--weight_save', dest='weight_save_dir', required=False, default=None, help='Path to save weights of models', type=str)
    parser.add_argument('-l', '--weight_load', dest='weight_load_dir', required=False, default=None, help='Path to load weights of models', type=str)
    parser.add_argument('--image_dir', dest='image_dir', required=False, default='./samples', help='Path to load images. See GitHub/samples', type=str)

    args = parser.parse_args()

    return args


def getArgParserForInference():

    parser = argparse.ArgumentParser(description='Overview of the system')

    parser.add_argument('-w', "--width", dest='w', required=False, default=512, help='width of input images', type=int)
    parser.add_argument("--height", dest='h', required=False, default=512, help='height of input images', type=int)
    parser.add_argument("-c", '--channel', dest='channel', required=False, default=1, help='channels of input images', type=int)
    parser.add_argument('-o', '--output', dest='output', required=False, default=None, help='Path to save predictions by models', type=str)
    parser.add_argument('-l', '--weight_load', dest='weight_load_dir', required=False, default=None, help='Path to load weights of models', type=str)
    parser.add_argument('-s', '--source', dest='source', required=False, default='./samples/mri', help='Path to load images. See GitHub/samples', type=str)

    args = parser.parse_args()

    return args


