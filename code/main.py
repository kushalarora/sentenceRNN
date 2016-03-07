import argparse
from sentRNN import train, test
MODE_TRAIN = 0
MODE_TEST = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int,
                        help='Run mode: \n0 => Train, 1 => Test, 2=>BOTH',
                        default=0)
    parser.add_argument('--train', type=str, help='training text file',
                        default='../data/ptb.train.txt')
    parser.add_argument('--valid', type=str, help='validation text file',
                        default='../data/ptb.valid.txt')
    parser.add_argument('--test', type=str, help='test text file',
                        default='../data/ptb.test.txt')
    parser.add_argument('--n', type=int, help='dimension of embedding space', default=100)
    parser.add_argument('--epochs', type=int, help='dimension of embedding space', default=20)
    parser.add_argument('--l', type=float, help='learning rate', default=0.1)
    parser.add_argument('--trnd', type=int, help='training distortion level', default=10)
    parser.add_argument('--tstd', type=int, help='test distortion level', default=30)
    parser.add_argument('--l2reg', type=float, help='lambda l2 reg value', default=0.001)
    parser.add_argument('--pct', type=float, help='Percentage of training data to use', default=100)

    args = parser.parse_args()
    N = args.n

    if (args.mode == MODE_TRAIN):
        train(args.l, args.n, args.epochs, args.train,
                args.trnd, args.tstd, args.valid, args.l2reg, args.pct)


    if args.mode == MODE_TEST:
        test('best_model.pkl', args.tstd, args.test)


