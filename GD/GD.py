from math import sqrt, sin
import random

class GD():
    def __init__(self):
        self.alpha = 0.005

    # check if grad (list type) consists from zeroes
    def zeroes(self, grad, eps = 1e-5):
        fl = True
        for g in grad:
            if abs(g) > eps:
                fl = False
                break
        return fl

    def calculate_deriv(self, f, weights, example, f_num):
        eps = 1e-5
        weights_eps = weights.copy()
        weights_eps[f_num] += eps
        return (f(weights_eps, example) - f(weights, example)) / eps

    def calculate_err(self, f, weights, examples, results):
        errors = [abs(results[i] - f(weights, examples[i])) for i in range(len(examples))]
        return sum(errors) / len(errors)

    def train(self, f, dataset, print_fl = False):
        examples, results = dataset
        features_num = len(examples[0])
        iters = 0
        weights = [random.random() for i in range(features_num)]
        print('Initial weights: ', weights)
        grad = self.calculate_grad(f, weights, examples, results)
        prev_err = self.calculate_err(f, weights, examples, results)
        while not (self.zeroes(grad)):
            iters += 1
            weights = [weights[i] - self.alpha * grad[i] for i in range(features_num)]
            grad = self.calculate_grad(f, weights, examples, results)
            err = self.calculate_err(f, weights, examples, results)
            if (err > prev_err):
                self.alpha *= 0.9
            else:
                self.alpha *= 1.1
            prev_err = err
            if print_fl:
                print(weights, err)
        print('Iterations: ', iters)
        return weights

class GD_MSE(GD):
    def calculate_grad(self, f, weights, examples, results):
        features_num = len(examples[0])
        examples_num = len(examples)
        grad = [0] * features_num
        for weight_num in range(features_num):
            deriv_sum = 0
            for i in range(examples_num):
                deriv_sum += self.calculate_deriv(f, weights, examples[i], weight_num) * (f(weights, examples[i]) - results[i])
            grad[weight_num] = (2 / examples_num) * deriv_sum
        return grad

class GD_MSE_stohastic(GD):
    def calculate_grad(self, f, weights, examples, results):
        features_num = len(examples[0])
        examples_num = len(examples)
        grad = [0] * features_num
        i = random.randint(0, examples_num - 1)
        for weight_num in range(features_num):
            deriv_sum = self.calculate_deriv(f, weights, examples[i], weight_num) * (f(weights, examples[i]) - results[i])
            grad[weight_num] = (2 / examples_num) * deriv_sum
        return grad

def gen_data(f, W, A, n):
    for i in range(n):
        A = [random.random() * 10 for j in range(len(W))]
        print(f(W, A), A)

def read_data(file_path):
    f = open(file_path, 'r')
    examples = []
    results = []
    for line in f:
        if line.startswith('#'):  # comment
            continue
        line_parts = line.split(' ')
        results.append(float(line_parts[0]))
        examples.append([float(feature) for feature in line_parts[1:]])
    return (examples, results)

# w1*sin(a1) + w2*sin(a2)...
def f1(weights, features):
    res = 0
    for i in range(len(weights)):
        res += weights[i] * sin(features[i])
    return res

# w1*a1 + w2*a2...
def f2(weights, features):
    res = 0
    for i in range(len(weights)):
        res += weights[i] * features[i]
    return res

dataset = read_data('data.txt')
gd_mse = GD_MSE_stohastic()
W = gd_mse.train(f2, dataset)
print('Result: ', W)
