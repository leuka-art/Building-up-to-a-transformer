import numpy as np

from Autograd import Tensor


np.random.seed(0)

def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient using central difference.
    """

    grad = np.zeros_like(x)

    iterator = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"]
    )

    while not iterator.finished:
        idx = iterator.multi_index

        original = x[idx]

        x[idx] = original + eps
        plus = f(x)

        x[idx] = original - eps
        minus = f(x)

        x[idx] = original

        grad[idx] = (plus - minus) / (2 * eps)

        iterator.iternext()

    return grad


def gradcheck(function, tensor, eps=1e-5, tolerance=1e-6):
    """
    Compare analytical autograd gradient with numerical gradient.
    
    function:
        function taking a Tensor and returning a scalar Tensor

    tensor:
        Tensor to check gradients for
    """

    # analytical gradient
    tensor.grad = np.zeros_like(tensor.data)

    output = function(tensor)
    output.backward()

    analytical_grad = tensor.grad.copy()


    # numerical gradient
    numerical_grad = numerical_gradient(
        lambda x: function(
            Tensor(x, requires_grad=True)
        ).data,
        tensor.data.copy(),
        eps
    )


    # relative error
    error = np.linalg.norm(
        analytical_grad - numerical_grad
    ) / (
        np.linalg.norm(analytical_grad)
        +
        np.linalg.norm(numerical_grad)
        +
        1e-12
    )

    return error < tolerance, error

def run_test(name, fn, x, tolerance=1e-6):
    try:
        passed, error = gradcheck(fn, x, tolerance=tolerance)

        status = "PASS" if passed else "FAIL"

        print(f"{status:<5} {name:<35} error={error:.3e}")

    except Exception as e:
        print(f"FAIL  {name:<35} error={e}")

def test_all():
    # -------------------------
    # Basic operations
    # -------------------------
    x = Tensor(np.random.randn(3, 3), requires_grad=True)

    run_test(
        "add scalar",
        lambda x: (x + 2).sum(),
        x
    )

    run_test(
        "multiply scalar",
        lambda x: (x * 3).sum(),
        x
    )

    run_test(
        "subtract",
        lambda x: (x - 4).sum(),
        x
    )

    run_test(
        "divide scalar",
        lambda x: (x / 2).sum(),
        x
    )

    run_test(
        "power",
        lambda x: (x ** 3).sum(),
        x
    )

    run_test(
        "exp",
        lambda x: x.exp().sum(),
        x
    )

    run_test(
        "log",
        lambda x: (x + 1e-3).log().sum(),
        x+1
    )

    run_test(
        "relu",
        lambda x: x.ReLU().sum(),
        x
    )


    # -------------------------
    # Broadcasting
    # -------------------------

    x = Tensor(np.random.randn(3, 4), requires_grad=True)

    run_test(
        "broadcast scalar add",
        lambda x: (x + 2).sum(),
        x
    )


    v = Tensor(np.random.randn(4), requires_grad=True)

    run_test(
        "broadcast vector",
        lambda v: (x + v).sum(),
        v
    )


    row = Tensor(np.random.randn(1, 4), requires_grad=True)

    run_test(
        "broadcast row vector",
        lambda row: (x + row).sum(),
        row
    )


    col = Tensor(np.random.randn(3, 1), requires_grad=True)

    run_test(
        "broadcast column vector",
        lambda col: (x + col).sum(),
        col
    )


    high = Tensor(np.random.randn(1, 3, 4), requires_grad=True)

    run_test(
        "broadcast higher rank",
        lambda high: (high + 2).sum(),
        high
    )


    run_test(
        "multiply broadcast vector",
        lambda v: (x * v).sum(),
        v
    )


    # -------------------------
    # Reductions
    # -------------------------

    x = Tensor(np.random.randn(3, 4), requires_grad=True)

    run_test(
        "sum all",
        lambda x: x.sum(),
        x
    )

    run_test(
        "sum axis0",
        lambda x: x.sum(axis=0).sum(),
        x
    )

    run_test(
        "sum axis1",
        lambda x: x.sum(axis=1).sum(),
        x
    )

    run_test(
        "sum keepdims",
        lambda x: x.sum(axis=1, keepdims=True).sum(),
        x
    )


    run_test(
        "mean",
        lambda x: x.mean(),
        x
    )


    run_test(
        "mean axis",
        lambda x: x.mean(axis=1).sum(),
        x
    )


    # -------------------------
    # Shape operations
    # -------------------------

    run_test(
        "reshape",
        lambda x: x.reshape((4, 3)).sum(),
        x
    )


    run_test(
        "swap last axes",
        lambda x: x.transpose(-1, -2).sum(),
        x
    )


    # -------------------------
    # Matrix multiplication
    # -------------------------

    A = Tensor(
        np.random.randn(3, 4),
        requires_grad=True
    )

    B = Tensor(
        np.random.randn(4, 5),
        requires_grad=True
    )


    run_test(
        "matrix multiply",
        lambda A: (A @ B).sum(),
        A
    )


    v = Tensor(
        np.random.randn(1,4),
        requires_grad=True
    )

    run_test(
        "vector matrix multiply",
        lambda v: (v @ B).sum(),
        v
    )