"""Tests for FitConfiguration.tie_ions (tied v and/or b across ions)."""
import warnings

import numpy as np
import pytest

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.unified_results import UnifiedResults
from rbvfit import vfit_mcmc as mc

Z = 2.5429
OI, SIII = [1302.1685], [1304.3702]


def make_config(nc=2, v=True, b=False, extra_ion=False):
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=nc)
    if extra_ion:
        c.add_system(z=Z, ion='CII', transitions=[1334.5323], components=1)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=nc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c.tie_ions(z=Z, ions=['OI', 'SiII'], v=v, b=b)
    return c


def synthetic(config, truth, fwhm='2.5', seed=1):
    wave = np.arange(1298.0, 1308.5, 0.07) * (1 + Z)
    model = VoigtModel(config, FWHM=fwhm)
    rng = np.random.default_rng(seed)
    flux = model.evaluate(truth, wave) + rng.normal(0, 0.02, len(wave))
    return {'A': {'model': model, 'wave': wave, 'flux': flux,
                  'error': np.full(len(wave), 0.02)}}


# theta order for make_config(nc=2):
# [N_OI1, N_OI2, N_Si1, N_Si2, b_OI1, b_OI2, b_Si1, b_Si2, v_OI1, v_OI2, v_Si1, v_Si2]
TRUTH = np.array([14.5, 14.2, 13.8, 13.5, 20., 15., 30., 25., -100., 50., -100., 50.])
LB = np.array([12.] * 4 + [5.] * 4 + [-300.] * 4)
UB = np.array([17.] * 4 + [80.] * 4 + [300.] * 4)


# ---------------- configuration ----------------

def test_tie_source_v_only():
    ts = ParameterManager(make_config()).get_tie_source()
    expected = np.arange(12)
    expected[10:12] = [8, 9]
    assert np.array_equal(ts, expected)


def test_tie_source_v_and_b():
    ts = ParameterManager(make_config(b=True)).get_tie_source()
    expected = np.arange(12)
    expected[6:8] = [4, 5]
    expected[10:12] = [8, 9]
    assert np.array_equal(ts, expected)


def test_tie_source_skips_untied_ion_between():
    # layout: OI(2), CII(1), SiII(2) -> 5 components
    ts = ParameterManager(make_config(extra_ion=True)).get_tie_source()
    expected = np.arange(15)
    expected[13:15] = [10, 11]
    assert np.array_equal(ts, expected)


def test_tie_source_multiple_systems():
    c = FitConfiguration()
    c.add_system(z=0.3, ion='MgII', transitions=[2796.35, 2803.53], components=1)
    c.add_system(z=Z, ion='OI', transitions=OI, components=2)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=2)
    c.tie_ions(z=Z, ions=['OI', 'SiII'])
    ts = ParameterManager(c).get_tie_source()
    # 5 components: MgII(1), OI(2), SiII(2); v block starts at 10
    expected = np.arange(15)
    expected[13:15] = [11, 12]
    assert np.array_equal(ts, expected)


def test_b_only_warns():
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=2)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=2)
    with pytest.warns(UserWarning, match='without v'):
        c.tie_ions(z=Z, ions=['OI', 'SiII'], v=False, b=True)


@pytest.mark.parametrize('kwargs, match', [
    (dict(ions=['OI', 'SiII'], v=False, b=False), 'v=True and/or b=True'),
    (dict(ions=['OI']), 'at least two'),
    (dict(ions=['OI', 'OI']), 'Duplicate'),
    (dict(ions=['OI', 'CIV']), 'not found'),
])
def test_tie_errors(kwargs, match):
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=2)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=2)
    with pytest.raises(ValueError, match=match):
        c.tie_ions(z=Z, **kwargs)


def test_tie_requires_same_system_and_counts():
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=2)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=3)
    with pytest.raises(ValueError, match='No system'):
        c.tie_ions(z=1.0, ions=['OI', 'SiII'])
    with pytest.raises(ValueError, match='equal component counts'):
        c.tie_ions(z=Z, ions=['OI', 'SiII'])


def test_ion_in_one_group_only():
    c = make_config()
    c.add_system(z=Z, ion='CII', transitions=[1334.5323], components=2)
    with pytest.raises(ValueError, match='already in a tie group'):
        c.tie_ions(z=Z, ions=['CII', 'OI'])


def test_update_components_propagates_to_group():
    c = make_config(nc=2)
    c.update_system_components(Z, 'SiII', 4)
    assert [g.components for g in c.systems[0].ion_groups] == [4, 4]
    assert len(ParameterManager(c).get_tie_source()) == 24


def test_serialize_roundtrip():
    c = make_config(b=True)
    c2 = FitConfiguration.deserialize(c.serialize())
    assert c2.systems[0].ties == c.systems[0].ties
    assert np.array_equal(ParameterManager(c2).get_tie_source(),
                          ParameterManager(c).get_tie_source())


def test_untied_serialization_unchanged():
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=1)
    assert 'ties' not in c.serialize()


def test_apply_ties():
    theta = TRUTH.copy()
    theta[10:12] = [0., 0.]
    out = ParameterManager(make_config()).apply_ties(theta)
    assert np.array_equal(out[10:12], out[8:10])


# ---------------- fitting ----------------

def test_vfit_free_dimension_and_initial_ties():
    data = synthetic(make_config(), TRUTH)
    theta = TRUTH.copy()
    theta[10] += 5.
    with pytest.warns(UserWarning, match='Initial guesses of tied'):
        f = mc.vfit(data, theta, LB, UB)
    assert f.ndim == 10 and f.tied
    assert f.theta[10] == f.theta[8]


def test_lnprob_full_and_free_agree():
    data = synthetic(make_config(), TRUTH)
    f = mc.vfit(data, TRUTH, LB, UB)
    assert f.lnprob(TRUTH) == f._lnprob_free(TRUTH[f.free_mask])


def test_tied_bounds_ignored():
    # a tied entry outside its own bounds must not make the posterior -inf
    data = synthetic(make_config(), TRUTH)
    lb = LB.copy()
    lb[10] = -50.            # SiII c1 v bound excludes OI c1 v = -100
    theta = TRUTH.copy()
    theta[10] = 0.
    with pytest.warns(UserWarning):
        f = mc.vfit(data, theta, lb, UB)
    assert np.isfinite(f.lnprob(f.theta))


@pytest.mark.parametrize('b', [False, True])
def test_optimize_and_quick_fit_recover_truth(b):
    truth = TRUTH.copy()
    if b:
        truth[6:8] = truth[4:6]
    config = make_config(b=b)
    data = synthetic(config, truth)
    guess = truth + np.r_[[0.2] * 4, [3.] * 4, [8.] * 4]
    guess = ParameterManager(config).apply_ties(guess)
    f = mc.vfit(data, guess, LB, UB)
    ts = f.tie_source

    opt = f.optimize_guess(f.theta)
    assert np.array_equal(opt, opt[ts])
    assert np.allclose(opt[8:10], truth[8:10], atol=5.)

    best, err = f.fit_quick(verbose=False)
    assert len(best) == len(err) == 12
    assert np.array_equal(best, best[ts]) and np.array_equal(err, err[ts])


@pytest.mark.parametrize('sampler', ['emcee', 'zeus'])
def test_mcmc_and_unified_results(sampler, tmp_path):
    data = synthetic(make_config(), TRUTH)
    f = mc.vfit(data, TRUTH, LB, UB, no_of_Chain=24, no_of_steps=150, sampler=sampler)
    np.random.seed(0)
    f.runmcmc(optimize=False, verbose=False, use_pool=False, progress=False)

    assert f.samples.shape[1] == 12
    assert np.array_equal(f.samples[:, 10:12], f.samples[:, 8:10])
    assert f.best_theta[10] == f.best_theta[8]
    assert f.sampler.get_chain().shape[-1] == 12

    u = UnifiedResults(f)
    assert u.samples.shape[1] == 12 and u.chain.shape[-1] == 12
    u.save(tmp_path / 'tied.h5')
    u2 = UnifiedResults.load(tmp_path / 'tied.h5')
    assert np.array_equal(u2.best_fit, u.best_fit)


def test_mcmc_with_pool():
    data = synthetic(make_config(), TRUTH)
    f = mc.vfit(data, TRUTH, LB, UB, no_of_Chain=24, no_of_steps=120)
    f.runmcmc(optimize=False, verbose=False, use_pool=True, progress=False)
    assert np.array_equal(f.samples[:, 10:12], f.samples[:, 8:10])


def test_multi_instrument_ties():
    config = make_config()
    d = synthetic(config, TRUTH)
    d['B'] = dict(d['A'], model=VoigtModel(config, FWHM='1.5'))
    f = mc.vfit(d, TRUTH, LB, UB)
    assert f.tied and f.ndim == 10


def test_multi_instrument_tie_mismatch():
    d = synthetic(make_config(), TRUTH)
    c = FitConfiguration()
    c.add_system(z=Z, ion='OI', transitions=OI, components=2)
    c.add_system(z=Z, ion='SiII', transitions=SIII, components=2)
    d['B'] = dict(d['A'], model=VoigtModel(c, FWHM='1.5'))
    with pytest.raises(ValueError, match='same parameter ties'):
        mc.vfit(d, TRUTH, LB, UB)
