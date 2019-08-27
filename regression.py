# refer to: https://github.com/krasserm/bayesian-machine-learning/blob/master/bayesian_neural_networks.ipynb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

tf.enable_eager_execution()


def generate_data(data_size=32, noise=1.0):
    """从正弦曲线构造数据，并添加高斯噪声
    """
    x = np.linspace(-0.5, 0.5, data_size).reshape(-1, 1)
    y = 10 * np.sin(2 * np.pi * x) + np.random.randn(*x.shape) * noise
    return x, y


class DenseVariational(tf.keras.layers.Layer):
    def __init__(self, output_dim, prior_sigma_1, prior_sigma_2, prior_pi):
        super(DenseVariational, self).__init__()
        self.output_dim = output_dim
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_sigma = np.sqrt(self.prior_pi * self.prior_sigma_1 ** 2 +
                                   (1 - self.prior_pi) * self.prior_sigma_2 ** 2)

    def build(self, input_shape):
        self.kernel_mu = self.add_variable(
            "kernel_mu", shape=[input_shape[-1], self.output_dim],
            initializer=tf.keras.initializers.normal(stddev=self.prior_sigma), trainable=True)

        self.kernel_rho = self.add_variable(
            "kernel_sigma", shape=[input_shape[-1], self.output_dim],
            initializer=tf.keras.initializers.constant(0.0), trainable=True)

        self.bias_mu = self.add_variable(
            "bias_mu", shape=[self.output_dim],
            initializer=tf.keras.initializers.normal(stddev=self.prior_sigma), trainable=True)

        self.bias_rho = self.add_variable(
            "bias_sigma", shape=[self.output_dim],
            initializer=tf.keras.initializers.constant(0.0), trainable=True)

    def kl_loss_fn(self, w, mu, sigma):
        # log(q(w|θ))
        variational_dist = tfp.distributions.Normal(mu, sigma)
        prob1 = variational_dist.log_prob(w)
        # log(p(w))
        prior_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        prior_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        prob2 = tf.log(self.prior_pi * prior_1_dist.prob(w) + (1.0-self.prior_pi) * prior_2_dist.prob(w))
        return tf.reduce_mean(prob1 - prob2)

    def call(self, inp):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        # kl loss compute
        kl_loss_kernel = self.kl_loss_fn(kernel, self.kernel_mu, kernel_sigma)
        kl_loss_bias = self.kl_loss_fn(bias, self.bias_mu, bias_sigma)
        kl_loss = kl_loss_kernel + kl_loss_bias

        return tf.matmul(inp, kernel) + bias, kl_loss


class NNVariational(tf.keras.Model):
    def __init__(self, prior_trainable=True):
        super(NNVariational, self).__init__()
        prior_sigma_1 = 1.0  # prior is a mixture distribution
        prior_sigma_2 = 0.1
        prior_pi = 0.2
        # The prior
        self.sigma_1 = tf.Variable(prior_sigma_1, trainable=prior_trainable)
        self.sigma_2 = tf.Variable(prior_sigma_2, trainable=prior_trainable)
        self.pi = tf.Variable(prior_pi, trainable=prior_trainable)
        # dense
        self.dense1 = DenseVariational(20, self.sigma_1, self.sigma_2, self.pi)
        self.dense2 = DenseVariational(20, self.sigma_1, self.sigma_2, self.pi)
        self.dense3 = DenseVariational(1,  self.sigma_1, self.sigma_2, self.pi)

    def call(self, x):
        x, kl_loss_1 = self.dense1(x)
        x = tf.nn.relu(x)
        x, kl_loss_2 = self.dense2(x)
        x = tf.nn.relu(x)
        y, kl_loss_3 = self.dense3(x)

        kl_loss_total = kl_loss_1 + kl_loss_2 + kl_loss_3
        # print
        # print("kl_loss_dense1:", kl_loss_1.numpy(), "kl_loss_dense2:", kl_loss_2.numpy(), "kl_loss_dense3:", kl_loss_3.numpy())
        # print("Dense1: prior_sigma1:", self.dense1.prior_sigma_1.numpy(), ", prior_sigma2:", self.dense1.prior_sigma_2.numpy())
        # print("Dense2: prior_sigma1:", self.dense2.prior_sigma_1.numpy(), ", prior_sigma2:", self.dense2.prior_sigma_2.numpy())
        # print("Dense3: prior_sigma1:", self.dense3.prior_sigma_1.numpy(), ", prior_sigma2:", self.dense3.prior_sigma_2.numpy())

        return y, kl_loss_total


def neg_log_likelihood(y_obs, y_pred):
    dist = tfp.distributions.Normal(loc=y_pred, scale=1.0)
    return -1. * tf.reduce_mean(dist.log_prob(y_obs))


def train(m, x, y, opt, kl_weight=1.0):
    for epoch in range(1000):
        with tf.GradientTape() as tape:
            predict, kl_loss_total = m(x)
            data_loss = neg_log_likelihood(tf.convert_to_tensor(y, tf.float32), predict)
            loss = data_loss + kl_weight * kl_loss_total
        gradients = tape.gradient(loss, m.trainable_variables)
        opt.apply_gradients(zip(gradients, m.trainable_variables))
        if epoch % 10 == 0:
            print(epoch, data_loss.numpy(), kl_weight*kl_loss_total.numpy())


# train
model = NNVariational()
test = model(tf.convert_to_tensor(np.random.random((10, 1)), tf.float32))
model.summary()
optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
X, Y = generate_data()
train(model, tf.convert_to_tensor(X, tf.float32), Y, optimizer)

# test
X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []

for i in tqdm.tqdm(range(500)):
    y_pred = model(tf.convert_to_tensor(X_test, tf.float32))[0].numpy()
    y_pred_list.append(y_pred)

y_preds = np.concatenate(y_pred_list, axis=1)   # (1000, 500)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
plt.scatter(X, Y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.legend()
plt.savefig("pic-kl-1.jpg")
plt.show()
