import argparse

parser = argparse.ArgumentParser(description="args for training the neural network")
parser.add_argument("--seed", type=int, default = 1, help="seed")
parser.add_argument("--n_total", type=int, default = 100, help="number of neurons")
parser.add_argument("-batchsize", type=int, default = 1, help="batch size")
parser.add_argument("-epochs", type=int, default = 10, help="number of epochs during training")

parser.add_argument("--noise_common", type=float, default = 0.1, help="How much noise in the data")
parser.add_argument("--noise_group", type=float, default = 0.1, help="How much noise in the data")
parser.add_argument("--noise_independent", type=float, default = 0.1, help="How much noise in the data")

parser.add_argument("--add_const", type=float, default=0.0, help="Add constant to data (to get mean effects)")
parser.add_argument("--p", type=float, default=0.99, help="How much to lesion the last year")
parser.add_argument("--p_over", type=float, default=0.5, help="probability of over-representation")
parser.add_argument("-phase", type=str, default = "2", help="phases 1,2, or 3")
parser.add_argument("-optimization", type=str, default = "Adam", help="Adam, SGD, or others")
parser.add_argument("-criterion", type=str, default = "BCE", help="MSE or BCE?")
parser.add_argument("-centering", type=bool, default = False, help="to center or not to center")
parser.add_argument("-learning_rule", type=str, default = "Oja", help="what learning rule to use for training: Hebbian/Oja/3-factor Hebbian")

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_total
batchsize = args.batchsize
n_epochs = args.epochs
optimize = args.optimization
crit = args.criterion

gain_to_add = args.add_const
p = args.p
p_over = args.p_over

noise_common = args.noise_common
noise_group = args.noise_group
noise_independent = args.noise_independent

phase = args.phase
centering = args.centering

learning_rule = args.learning_rule
