"""Autograd is checked against numerical gradients: if these pass, the math is right."""
import numpy as np
import pytest

from nlpbasics import data, neural, text
from nlpbasics.autograd import Adam, Tensor, concat, cross_entropy
from nlpbasics.embeddings import SkipGram


def gradcheck(f, *arrays, h=1e-6, tol=1e-5):
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    f(*tensors).backward()
    for tensor, array in zip(tensors, arrays):
        for index in np.ndindex(array.shape):
            base = array[index]
            array[index] = base + h
            high = f(*[Tensor(a) for a in arrays]).item()
            array[index] = base - h
            low = f(*[Tensor(a) for a in arrays]).item()
            array[index] = base
            assert tensor.grad[index] == pytest.approx((high - low) / (2 * h), abs=tol)


rng = np.random.default_rng(0)
A, B, C = rng.normal(size=(3, 2)), rng.normal(size=(2, 3)), rng.normal(size=3)
POS = rng.uniform(0.5, 2.0, size=(3, 2))


@pytest.mark.parametrize("f,args", [
    (lambda a, c: ((a + c[None, :2]) * a).sum(), (A, C)),         # broadcasting
    (lambda a, c: (a - 1.5 * c[:2]).exp().sum(), (A, C)),
    (lambda a, b: ((a @ b) ** 2.0).sum(), (A, B)),
    (lambda p: (p.log() + p.sqrt() + 1.0 / p).sum(), (POS,)),
    (lambda a: a.tanh().sum() + a.sigmoid().sum() + (a * 3.0).relu().sum(), (A,)),
    (lambda a: a.softmax(-1).log().sum(), (A,)),
    (lambda a: (a.log_softmax(0) * a).sum(), (A,)),
    (lambda a: a.reshape(2, 3).transpose(1, 0).mean(axis=1).sum(), (A,)),
    (lambda a: a[[0, 0, 2], :].sum() + (a[1:, 1] ** 2.0).sum(), (A,)),
    (lambda a, b: concat([a, b.transpose(1, 0)], axis=1).sum(axis=0).mean(), (A, B)),
    (lambda a: a.masked_fill(np.array([[True, False]] * 3), -3.0).exp().sum(), (A,)),
    (lambda a: cross_entropy(a, np.array([0, 1, 1])), (A,)),
    (lambda a: cross_entropy(a, np.array([0, 9, 1]) % 2, ignore_index=1), (A,)),
])
def test_gradients_match_numerical(f, args):
    gradcheck(f, *[a.copy() for a in args])


def test_reused_node_accumulates():
    x = Tensor(3.0, requires_grad=True)
    (x * x + x).backward()
    assert x.grad.item() == 7.0


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(20000):
        y = y + 0.0
    y.backward()
    assert x.grad.item() == 1.0


def test_constants_build_no_graph():
    out = Tensor(1.0) * Tensor(2.0)
    assert not out.requires_grad and out._parents == ()


@pytest.mark.parametrize("make", [
    lambda: neural.WindowMLP(7, context=2, dim=3, hidden=4),
    lambda: neural.RNNLM(7, dim=3, hidden=4, cell="elman"),
    lambda: neural.RNNLM(7, dim=3, hidden=4, cell="gru"),
    lambda: neural.TinyTransformer(7, max_len=5, d_model=8, heads=2, layers=1),
])
def test_model_parameter_gradients(make):
    """Differentiate each whole model with respect to a sample of its parameters."""
    model = make()
    ids = np.array([[2, 3, 4, 5]])
    def loss():
        return cross_entropy(model(ids[:, :-1]), ids[:, 1:])
    loss().backward()
    assert model.parameters()
    for p in model.parameters():
        index = tuple(s // 2 for s in p.shape)
        analytic = p.grad[index] if p.grad is not None else 0.0
        base, h = p.data[index], 1e-5
        p.data[index] = base + h
        high = loss().item()
        p.data[index] = base - h
        low = loss().item()
        p.data[index] = base
        assert analytic == pytest.approx((high - low) / (2 * h), abs=1e-5)


def test_causal_mask_blocks_the_future():
    model = neural.TinyTransformer(10, max_len=8, d_model=16, heads=2, layers=2)
    a = np.array([[1, 2, 3, 4, 5, 6]])
    b = a.copy()
    b[0, 4:] = [9, 9]
    assert np.allclose(model(a).data[0, :4], model(b).data[0, :4])
    assert not np.allclose(model(a).data[0, 4:], model(b).data[0, 4:])


def test_attention_weights_are_distributions():
    model = neural.TinyTransformer(10, max_len=8, d_model=16, heads=2, layers=1)
    weights = model.attention_maps([1, 2, 3, 4])[0]
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert np.allclose(np.triu(weights, 1), 0.0, atol=1e-8)


def test_adam_minimizes_a_quadratic():
    x = Tensor(np.array([5.0, -3.0]), requires_grad=True)
    opt = Adam([x], lr=0.2, clip_norm=None)
    for _ in range(300):
        loss = (x ** 2.0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.abs(x.data).max() < 1e-2


def test_transformer_learns_long_range_agreement():
    train = data.agreement_corpus(2000, 0)
    test = data.agreement_corpus(100, 1, holdout=True)
    vocab = text.Vocabulary().fit(train)
    width = max(map(len, train + test))
    X = np.array([vocab.encode(s) + [0] * (width - len(s)) for s in train])
    model = neural.TinyTransformer(len(vocab), max_len=width)
    neural.train_language_model(model, X, steps=500)
    from lessons.l09_neural import agreement_accuracy
    assert agreement_accuracy(model, vocab, test) > 0.95


def test_skipgram_loss_falls_and_groups_topics():
    model = SkipGram(dim=20, epochs=3).fit(data.embedding_corpus(2000))
    from nlpbasics.embeddings import KeyedVectors
    kv = KeyedVectors(model.W, model.vocab)
    assert model.losses[-1] < model.losses[0]
    assert kv.similarity("cat", "dog") > kv.similarity("cat", "paris")
