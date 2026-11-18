import numpy as np
import torch
import torch.nn as nn

WEIGHTS_DIR = "results/Oja_new_model/"
GD_WEIGHTS_DIR = "results/GD_new_model/"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _shuffle(features, y):
    """Shuffle samples once so the online rule sees them in random order.

    Returns features as a float32 tensor and labels as a flat numpy array.
    """
    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(y).detach().cpu().numpy().flatten()
    perm = np.random.permutation(features.shape[0])
    return features[torch.from_numpy(perm)], labels[perm]


def _init_weights(student_w_init, n_features):
    """Copy of the initial weights (the caller's tensor is never modified)."""
    if student_w_init is None:
        return torch.randn(1, n_features) * 0.01
    return torch.as_tensor(student_w_init).float().clone()


def _load_or_init_weights(phase, parameter_list, student_w_init, n_features):
    """Phases 2/3 load the previous phase's saved weights; phase 1 initialises."""
    if phase in ("2", "3"):
        path = WEIGHTS_DIR + _weights_filename(parameter_list, phase)
        return torch.as_tensor(torch.load(path)).float().squeeze()
    return _init_weights(student_w_init, n_features)


def _output(w, b, x, gain):
    """Student output, with the bias scaled by `gain`."""
    return torch.dot(w, x).item() + gain * b


def _bias_step(w, b, x, bias_lr):
    """Push the bias toward a target output of 0, using the current weights.

    Returns the new bias and the output (without gain) it was computed from.
    """
    y_bar = torch.dot(w, x).item() + b
    return b - bias_lr * y_bar, y_bar


def _hebbian_update(w, x, y_out, eta, target_norm=None):
    """In-place Hebbian update  dw = eta * y * x.

    If `target_norm` is given, w is rescaled to that norm after the update
    (explicit normalisation; Oja's rule is its first-order approximation).
    """
    w += eta * y_out * x
    if target_norm is not None:
        w *= target_norm / w.norm()


def _oja_update(w, x, y_out, eta, modulator=1.0):
    """In-place Oja update  dw = eta * M * (y * x - y^2 * w)."""
    w += eta * modulator * (y_out * x - y_out ** 2 * w)


def _oja_bias_step(w, b, x, gain, eta, bias_lr):
    """One online step: Oja update of w (in place), then a bias update.

    The Oja output includes the gain-scaled bias; the bias is then pushed
    toward a target output of 0 using the post-update output (without gain).
    Returns the new bias and that post-update output.
    """
    y_out = _output(w, b, x, gain)
    _oja_update(w, x, y_out, eta)
    return _bias_step(w, b, x, bias_lr)


def _targets(labels, flip):
    """Labels {0, 1} -> targets {-1, +1}, matching the sign used by _predict."""
    t = 2.0 * labels - 1.0
    return -t if flip else t


def _predict(outputs, flip):
    """Binary prediction in {0, 1} from the sign of the output."""
    outputs = np.asarray(outputs)
    return (outputs < 0 if flip else outputs > 0).astype(int)


def _window_accuracy(outputs, labels, end, flip, window=100):
    """Accuracy over the `window` samples just before index `end`."""
    preds = _predict(outputs[end - window:end], flip)
    return (preds == labels[end - window:end]).mean()


def _centering_bias(w, features):
    """Heuristic bias that centres the student's output over the dataset."""
    return -torch.mean(features @ w.reshape(-1))


def _weights_filename(params, phase, prefix="weights"):
    """Weights saved by the previous phase (phase 2 loads phase 1, 3 loads 2).

    Also checks that `phase` agrees with params["phase"] (if present), since
    the file name and the gain must refer to the same phase.
    """
    stored_phase = str(params.get("phase", phase))
    if stored_phase != phase:
        raise ValueError(f"phase={phase} but parameter_list['phase']={stored_phase}")
    p_over = 0.8 if phase == "2" else 0.5
    noise = "_".join(str(n) for n in params["noise_gp_co_ind"][:3])
    return (
        f"{prefix}_crit_{params['criterion']}_p_over_{p_over}_{params['W_init']}"
        f"_noise_gp_co_ind_{noise}_seed_{params['seed']}"
        f"_phase{int(phase) - 1}{params['centering']}.pt"
    )


# --------------------------------------------------------------------------- #
# Plain Hebbian
# --------------------------------------------------------------------------- #

def train_hebbian_unsupervised(features, y, student_w_init=None, add_bias=False,
                               phase="1", parameter_list=None,
                               n_epochs=300, eta=1e-8, bias_lr_scale=1000,
                               normalize=True, window=100, verbose=True):
    """Train a single linear student with the plain Hebbian rule dw = eta * y * x.

    Same setup as `train_oja_unsupervised` (shuffling, phases, gain, learned
    bias, accuracy windows). Pure Hebbian learning grows |w| without bound;
    with `normalize=True` w is rescaled to its initial norm after every step,
    which is the explicit normalisation that Oja's rule approximates.

    Returns (student_w, student_b, eta, acc_arr).
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)
    gain = 1 if phase == "2" else 0

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)
    target_norm = w.norm().item() if normalize else None
    b = 0.0
    bias_lr = bias_lr_scale * eta
    acc_arr = []

    if verbose:
        print("phase", phase)
        print("Student learning via Hebbian rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                y_out = _output(w, b, x, gain)
                _hebbian_update(w, x, y_out, eta, target_norm=target_norm)
                b, y_bar = _bias_step(w, b, x, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=False, window=window))

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr


# --------------------------------------------------------------------------- #
# Oja with a fixed gain on the bias
# --------------------------------------------------------------------------- #

def train_oja_unsupervised(features, y, student_w_init=None, add_bias=False,
                           phase="1", parameter_list=None,
                           n_epochs=300, eta=1e-8, bias_lr_scale=1000,
                           window=100, verbose=True):
    """Train a single linear student with Oja's rule plus a learned bias.

    Phase 1 and 3 use gain 0 (bias not fed into the Oja output); phase 2 uses
    gain 1. Phases 2 and 3 start from the weights saved by the previous phase
    (found via `parameter_list`); phase 1 starts from `student_w_init` or a
    small random vector.

    Returns (student_w, student_b, eta, acc_arr). If `add_bias` is True,
    student_b is the learned bias; otherwise it is the centering heuristic.
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)

    gain = 1 if phase == "2" else 0
    if verbose:
        print("phase", phase)

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)   # work on a flat view; same storage as the returned tensor
    b = 0.0
    bias_lr = bias_lr_scale * eta
    acc_arr = []

    if verbose:
        print("Student learning via Oja's Rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                b, y_bar = _oja_bias_step(w, b, x, gain, eta, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=False, window=window))

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr


# --------------------------------------------------------------------------- #
# Three-factor Oja (error-modulated)
# --------------------------------------------------------------------------- #

def _sample_error(y_out, label, target, error_type, flip):
    """Per-sample error: 0/1 mistake, or squared distance to the ±1 target."""
    if error_type == "binary":
        return float(_predict(y_out, flip) != label)
    if error_type == "squared":
        return (target - y_out) ** 2
    raise ValueError(f"error_type must be 'binary' or 'squared', got {error_type!r}")


def train_3factor_oja(features, y, student_w_init=None, add_bias=False,
                      phase="1", parameter_list=None, error_type="squared",
                      n_epochs=300, eta=1e-8, bias_lr_scale=1000, alpha=0.1,
                      flip=False, window=100, verbose=True):
    """Oja's rule gated by a third factor: dw = eta * M * (y * x - y^2 * w).

    The modulator compares the current sample's error with a running baseline
    (an exponential moving average with rate `alpha`):
        M = -(err - err_bar)
    so lower-than-usual error gives an Oja step and higher-than-usual error an
    anti-Oja step. Subtracting the baseline keeps the mean update near zero,
    as in reward-modulated Hebbian learning. No explicit normalisation: the
    Oja decay term keeps |w| near 1 on average (note that when M < 0 the decay
    term also flips sign, so |w| is only held on average, not at every step).

    error_type:
        "squared" - err = (t - y)^2 with targets t in {-1, +1}
        "binary"  - err = 1 if the sign prediction is wrong, else 0
                    (no squared error; same 0/1 error as the Adler variant)

    `flip` sets the sign convention (False: output > 0 predicts label 1, as in
    train_oja_unsupervised; True: output < 0 predicts 1, as in the Adler variant).
    Weights and bias are otherwise handled as in `train_oja_unsupervised`.

    Returns (student_w, student_b, eta, acc_arr, err_arr), where err_arr is the
    running error baseline recorded at each accuracy window.
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)
    targets = _targets(labels, flip)
    gain = 1 if phase == "2" else 0

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)
    b = 0.0
    bias_lr = bias_lr_scale * eta
    err_bar = None   # initialised from the first sample
    acc_arr, err_arr = [], []

    if verbose:
        print("phase", phase, "error_type", error_type)
        print("Student learning via three-factor Oja rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                y_out = _output(w, b, x, gain)
                err = _sample_error(y_out, labels[i], targets[i], error_type, flip)
                if err_bar is None:
                    err_bar = err
                modulator = -(err - err_bar)
                err_bar = (1 - alpha) * err_bar + alpha * err

                _oja_update(w, x, y_out, eta, modulator)
                b, y_bar = _bias_step(w, b, x, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=flip, window=window))
                    err_arr.append(err_bar)

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr, err_arr


# --------------------------------------------------------------------------- #
# Oja with an error-driven gain (latent phase oscillator)
# --------------------------------------------------------------------------- #

def _gain_from_latent(theta):
    """Map the latent phase to a gain in [0, 1]: 0 at theta=0, 1 at theta=pi."""
    return (1 - np.cos(theta)) / 2


def train_oja_unsupervised_adler(features, y, student_w_init=None, add_bias=False,
                                 phase="1", n_epochs=40, eta=1e-5, bias_lr_scale=1000,
                                 alpha=0.1, beta=0.01, A=2.5, gamma=10.0,
                                 phase3_bias=-1.2289, window=100, verbose=True):
    """Oja's rule where the bias gain is driven by the running error rate.

    The error rate is an exponential moving average (rate `alpha`) of per-sample
    mistakes. The gain signal 1 - 2|error - 0.5| peaks when the student is at
    chance, and drives a latent phase theta via
        theta += beta * (gamma * gain_signal - A * sin(2 theta)),
    whose stable points are theta=0 (gain 0) and theta=pi (gain 1).
    For bistability, need gamma*s_ok < A < gamma*s_fail.

    Phase 3 starts at theta=pi (gain 1) with zero error and bias `phase3_bias`.

    Returns (student_w, student_b, eta, acc_arr, gain_arr). If `add_bias` is
    True, student_b is the learned bias; otherwise it is the centering heuristic.
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)
    w = _init_weights(student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)

    if phase == "3":
        theta, error, b = np.pi, 0.0, phase3_bias
    else:
        theta, error, b = 0.0, 0.0, 0.0
    gain = _gain_from_latent(theta)
    bias_lr = bias_lr_scale * eta

    if verbose:
        print("eta", eta, w.mean().item())
        print("Student learning via Oja's Rule...")

    acc_arr, gain_arr = [], []
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                b, y_bar = _oja_bias_step(w, b, x, gain, eta, bias_lr)
                outputs.append(y_bar)

                # Error-driven gain dynamics
                mistake = float(_predict(y_bar, flip=True) != labels[i])
                error = (1 - alpha) * error + alpha * mistake
                gain_signal = 1 - 2 * abs(error - 0.5)
                theta += beta * (gamma * gain_signal - A * np.sin(2 * theta))
                theta %= 2 * np.pi
                gain = _gain_from_latent(theta)

                if i > 0 and i % window == 0:
                    acc = _window_accuracy(outputs, labels, i, flip=True, window=window)
                    acc_arr.append(acc)
                    gain_arr.append(gain)
                    if verbose:
                        print(f"i={i:5d}  acc={acc:.2f}  error={error:.3f}  "
                              f"gain_signal={gain_signal:.3f}  theta={theta:.3f}  "
                              f"gain={gain:.3f}  b={b:.4f}")

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr, gain_arr


# --------------------------------------------------------------------------- #
# Gradient-descent baseline (trains an nn.Linear-style network)
# --------------------------------------------------------------------------- #

def _gd_correct(out, y, criterion):
    """Boolean mask of correct predictions for MSE (0.5 threshold) or BCE-with-logits (0 threshold)."""
    out, y = out.reshape(-1), y.reshape(-1)
    if isinstance(criterion, nn.MSELoss):
        threshold = 0.5
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        threshold = 0.0
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    return ((out < threshold) & (y == 0)) | ((out > threshold) & (y == 1))


def _gd_accuracy(out, y, criterion):
    return _gd_correct(out, y, criterion).float().mean()


def _predict_dataset(network, dataloader, device):
    """Run the network over every batch; return (outputs, X, y) concatenated."""
    outs, xs, ys = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device)
            outs.append(network(X).reshape(-1))
            xs.append(X)
            ys.append(y.reshape(-1))
    return torch.cat(outs), torch.cat(xs), torch.cat(ys)


def train_model(network, train_dataloader, test_dataloader, optimizer, criterion,
                n_epochs, parameter_list, phase, device, log_every=5, verbose=True):
    """Train `network` with gradient descent, then evaluate on the test set.

    Phases 2 and 3 start from the weights saved by the previous phase. If
    parameter_list["W_init"] != "no_initW", 1 is then added to every weight
    (as in the original, this happens in every phase, including after loading).

    Returns (network, training_acc, valid_acc): training_acc has one entry per
    batch; valid_acc is computed over the whole test set.
    """
    phase = str(phase)
    network.float()

    if phase in ("2", "3"):
        path = GD_WEIGHTS_DIR + _weights_filename(parameter_list, phase, prefix="weights_GD")
        with torch.no_grad():
            network.weight.copy_(torch.load(path))

    if parameter_list["W_init"] != "no_initW":
        if verbose:
            print("init W", parameter_list["W_init"])
        with torch.no_grad():
            network.weight.add_(1)

    network.train()
    training_acc = []
    for _ in range(n_epochs):
        for bid, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device).float(), y.to(device).float().reshape(-1)

            optimizer.zero_grad()
            logits = network(X).reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc = _gd_accuracy(logits.detach(), y, criterion).item()
            training_acc.append(acc)
            if verbose and bid % log_every == 0:
                print(f"batch {bid:4d}  loss {loss.item():.4f}  accuracy {acc:.3f}")

    network.eval()
    out, _, y_test = _predict_dataset(network, test_dataloader, device)
    valid_acc = _gd_accuracy(out, y_test, criterion)
    if verbose:
        print("testing accuracy teacher:", valid_acc.item())
    return network, training_acc, valid_acc


def test_model_simple(network, test_dataloader, criterion, device, verbose=True):
    """Evaluate `network` on the whole test set.

    Returns (valid_acc, valid_loss, X, y), with X and y concatenated over all
    test batches.
    """
    network.eval()
    network.float()
    out, X, y = _predict_dataset(network, test_dataloader, device)
    valid_loss = criterion(out, y.float())
    valid_acc = _gd_accuracy(out, y, criterion)
    if verbose:
        print(f"VALID Loss {valid_loss:.4f}, Accuracy {valid_acc:.4f}")
    return valid_acc, valid_loss, X, y