"""Paper equations (1), (5)-(10), (13)-(16); SI units throughout."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import find_peaks

@dataclass(frozen=True)
class Config:
    radius_m: float = 0.017
    pitch_m: float = 0.005
    fiber_length_m: float = 4.277
    backbone_length_m: float = 0.200
    zones: int = 8
    strain_scale: float = 0.012
    frequency_modifier: float = 0.999
    phase_zero_rad: float = 0.0
    geometry_phi_offset_rad: float = 0.0
    gap_length_m: float = 0.017
    disk_length_m: float = 0.009
    gap_slope_strain_per_n: float = 16.845e-6
    gap_intercept_strain: float = 42.556e-6
    disk_slope_strain_per_n: float = 27.914e-6
    disk_intercept_strain: float = 35.526e-6
    threshold_strain: float = 50e-6
    prominence_strain: float = 20e-6
    min_peak_distance_m: float = 0.025
    noise_floor_strain: float = 1e-6
    tukey_c: float = 4.685
    curvature_min_per_m: float = 1e-4
    background_knot_spacing_m: float = 0.
    live_fit_window_m: tuple[float, float] | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'live_fit_window_m':
                continue
            if not np.isfinite(value):
                raise ValueError(f'{key} must be finite')
        for key in ('radius_m', 'pitch_m', 'fiber_length_m', 'backbone_length_m',
                    'strain_scale', 'frequency_modifier', 'gap_length_m', 'disk_length_m',
                    'gap_slope_strain_per_n', 'disk_slope_strain_per_n',
                    'noise_floor_strain', 'tukey_c'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.zones < 1 or int(self.zones) != self.zones:
            raise ValueError('zones must be a positive integer')
        if self.live_fit_window_m is not None:
            lower, upper = self.live_fit_window_m
            if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper <= self.fiber_length_m:
                raise ValueError('live_fit_window_m must be an ordered window in local fiber coordinates')
        for key in ('threshold_strain', 'prominence_strain', 'min_peak_distance_m',
                    'curvature_min_per_m', 'gap_intercept_strain', 'disk_intercept_strain',
                    'background_knot_spacing_m'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')

class HelicalFiberModel:
    def __init__(self, config=None):
        self.config = config or Config()

    @property
    def lam(self):
        c = self.config
        return np.hypot(c.radius_m, c.pitch_m/(2*np.pi))

    def coordinates(self, s):
        s = np.asarray(s, dtype=float)
        if s.ndim != 1 or not np.all(np.isfinite(s)) or len(s) == 0:
            raise ValueError('s must be a finite nonempty 1D array')
        if np.any(s < 0) or np.any(s > self.config.fiber_length_m + 1e-10):
            raise ValueError('s outside sensing fiber; subtract the lead-in coordinate first')
        return s

    def envelope(self, s):
        # Eq. (7) in fiber coordinates: equivalently use z=L*s/Lf on backbone.
        # sin(pi*N*s/Lf)*(-1)**(floor(N*s/Lf)+1) == -abs(sin(...)).
        s = self.coordinates(s)
        return -np.abs(np.sin(np.pi*self.config.zones*s/self.config.fiber_length_m))

    def design(self, s, fit_offset=True):
        s = self.coordinates(s)
        c = self.config
        q = c.frequency_modifier*s/self.lam + c.phase_zero_rad
        factor = -c.strain_scale*c.radius_m*self.envelope(s)
        x = np.column_stack((factor*np.sin(q), factor*np.cos(q)))
        if not fit_offset:
            return x
        spacing = c.background_knot_spacing_m
        if spacing and s[-1] > s[0]:
            knots = np.r_[[s[0]]*4, np.arange(s[0]+spacing, s[-1], spacing), [s[-1]]*4]
            return np.column_stack((x, BSpline.design_matrix(s, knots, 3).toarray()))
        return np.column_stack((x, np.ones(len(s))))

    def bending(self, s, kappa, phi, offset=0.):
        if not np.isfinite([kappa, phi, offset]).all() or kappa < 0:
            raise ValueError('finite kappa >= 0, phi and offset required')
        return self.design(s, False) @ np.array([kappa*np.cos(phi), kappa*np.sin(phi)]) + offset

    def zone(self, s):
        c = self.config
        z = float(s)*c.backbone_length_m/c.fiber_length_m
        return 'gap' if z % (c.gap_length_m+c.disk_length_m) < c.gap_length_m else 'disk'

    def calibration(self, s):
        name = self.zone(s)
        return (getattr(self.config, name+'_slope_strain_per_n'),
                getattr(self.config, name+'_intercept_strain'))

    def simulate(self, s, kappa, phi, contacts=(), noise_std=1e-6, seed=42, offset=0.):
        """Gaussian local peaks are a simulation assumption, NOT a paper constitutive law.
        contacts: dictionaries {s_m, force_n, width_m, sign(optional +/-1)}.
        """
        s = self.coordinates(s)
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        bend = self.bending(s, kappa, phi, offset)
        local = np.zeros_like(s)
        for event in contacts:
            pos, force, width = (event[k] for k in ('s_m', 'force_n', 'width_m'))
            sign = event.get('sign', 1)
            self.coordinates([pos])
            if not np.isfinite([force, width]).all() or width <= 0 or force < 0 or sign not in (-1, 1):
                raise ValueError('invalid contact force, width or sign')
            if force == 0:
                continue  # intercept is only valid on the calibrated contact branch
            slope, intercept = self.calibration(pos)
            local += sign*(slope*force+intercept)*np.exp(-0.5*((s-pos)/width)**2)
        return {'strain': bend+local+np.random.default_rng(seed).normal(0, noise_std, len(s)),
                'bending': bend, 'contact': local}

    def fit(self, s, strain, method='tukey', fit_offset=True, max_iter=100):
        """Exact linear reparameterization a=kappa*cos(phi), b=kappa*sin(phi).
        Tukey IRLS solves weighted least squares, without nonlinear phase local minima.
        NaN/Inf strain is masked, never replaced by zero. Coordinates remain ordered.
        """
        s = self.coordinates(s)
        y = np.asarray(strain, dtype=float)
        if y.shape != s.shape or np.any(np.diff(s) <= 0):
            raise ValueError('matching 1D arrays and strictly increasing s required')
        if method not in ('ols', 'tukey') or max_iter < 1:
            raise ValueError('method must be ols or tukey; max_iter >= 1')
        valid = np.isfinite(y)
        design = self.design(s, fit_offset)
        x, target = design[valid], y[valid]
        if len(target) < max(10, 3*x.shape[1]):
            raise ValueError('too few valid samples')
        scale_x = np.linalg.norm(x, axis=0)
        if np.any(scale_x < 1e-14):
            raise ValueError('unobservable design: envelope nodes or insufficient phase coverage')
        xn = x/scale_x
        if np.linalg.matrix_rank(xn) != x.shape[1] or np.linalg.cond(xn) > 1e8:
            raise ValueError('rank deficient or ill-conditioned design')
        beta = np.linalg.lstsq(xn, target, rcond=None)[0]
        weights = np.ones(len(target))
        converged = method == 'ols'
        iteration = 0
        if method == 'tukey':
            for iteration in range(1, max_iter+1):
                residual = target-xn@beta
                sigma = max(1.4826*np.median(np.abs(residual-np.median(residual))),
                            self.config.noise_floor_strain)
                u = residual/(self.config.tukey_c*sigma)
                weights = np.where(np.abs(u)<1, (1-u*u)**2, 0.)
                root = np.sqrt(weights)
                wx = xn*root[:, None]
                if np.linalg.matrix_rank(wx) != x.shape[1]:
                    raise ValueError('Tukey rejected too many samples to identify shape')
                new = np.linalg.lstsq(wx, target*root, rcond=None)[0]
                delta = np.max(np.abs(xn@(new-beta)))
                beta = new
                if delta < 1e-11:
                    converged = True
                    break
        coef = beta/scale_x
        kappa = float(np.hypot(*coef[:2]))
        phi = float(np.arctan2(coef[1], coef[0]))
        phase_valid = kappa >= self.config.curvature_min_per_m
        fitted = design@coef
        residual = y-fitted
        all_weights = np.zeros(len(s)); all_weights[valid] = weights
        warnings = []
        if not converged:
            warnings.append('IRLS did not converge within max_iter')
        if not phase_valid:
            warnings.append('Near-zero curvature: bending-plane angle and bent azimuth are unidentifiable')
        if np.mean(weights < .1) > .3:
            warnings.append('Many rejected samples: sparse-contact/PCC assumption may be invalid')
        if np.max(np.diff(s)) > (2*np.pi*self.lam/self.config.frequency_modifier)/4:
            warnings.append('Sparse sampling or large gaps: carrier and contact localization may be unreliable')
        summary = {'kappa_per_m': kappa, 'theta_deg': float(np.rad2deg(kappa*self.config.backbone_length_m)),
                   'phi_deg': float(np.rad2deg(phi)) if phase_valid else None,
                   'phi_rad': phi if phase_valid else None,
                   'offset_strain': float(np.mean(design[:, 2:] @ coef[2:])) if fit_offset else 0.,
                   'converged': converged, 'iterations': iteration, 'valid_samples': int(valid.sum()),
                   'rmse_strain': float(np.sqrt(np.mean(residual[valid]**2))),
                   'inlier_rmse_strain': float(np.sqrt(np.average(residual[valid]**2, weights=weights))),
                   'design_condition': float(np.linalg.cond(xn)), 'warnings': warnings}
        contacts = self.detect_contacts(s, residual, kappa, phi if phase_valid else 0.)
        return {'summary': summary, 'contacts': contacts, 'fitted': fitted,
                'bending': design[:, :2] @ coef[:2], 'residual': residual, 'weights': all_weights}

    def detect_contacts(self, s, residual, kappa=0., phi=0.):
        # Split on missing samples: never bridge an unmeasured interval.
        good = np.flatnonzero(np.isfinite(residual))
        runs = np.split(good, np.flatnonzero(np.diff(good)>1)+1)
        candidates = []
        for run in runs:
            if len(run)<3:
                continue
            peaks, _ = find_peaks(np.abs(residual[run]), height=self.config.threshold_strain,
                                 prominence=self.config.prominence_strain)
            candidates.extend(run[peaks].tolist())
        # Physical distance suppression also works on nonuniform coordinates.
        selected = []
        for i in sorted(candidates, key=lambda i: abs(residual[i]), reverse=True):
            if all(abs(s[i]-s[j]) >= self.config.min_peak_distance_m for j in selected):
                selected.append(i)
        results = []
        for i in sorted(selected):
            slope, intercept = self.calibration(s[i])
            force = float(max(0., (abs(residual[i])-intercept)/slope))
            results.append({'sample_index': int(i), 's_m': float(s[i]),
                'backbone_s_m': float(s[i]*self.config.backbone_length_m/self.config.fiber_length_m),
                'zone': self.zone(s[i]), 'peak_strain': float(residual[i]), 'force_n': force,
                'calibration_extrapolated': not 1. <= force <= 3.,
                'xyz_m': self.helix_xyz([s[i]], kappa, phi)[0].tolist()})
        return results

    def detect_contact_groups(self, s, residual, kappa=0., phi=0.):
        """Group consecutive above-threshold samples into measured surface traces."""
        active = np.flatnonzero(np.isfinite(residual) & (np.abs(residual) >= self.config.threshold_strain))
        groups = []
        for indices in np.split(active, np.flatnonzero(np.diff(active) > 1) + 1):
            if len(indices) < 2:
                continue
            locations = s[indices]
            values = residual[indices]
            groups.append({
                'sample_start_index': int(indices[0]),
                'sample_stop_index': int(indices[-1]),
                'sample_count': len(indices),
                's_m': locations.tolist(),
                'peak_strain': float(values[np.argmax(np.abs(values))]),
                'xyz_m': self.helix_xyz(locations, kappa, phi).tolist(),
            })
        return groups

    def _backbone_geometry(self, s, kappa, phi):
        s = self.coordinates(s)
        c = self.config
        phi = phi+c.geometry_phi_offset_rad
        l = s*c.backbone_length_m/c.fiber_length_m
        t = kappa*l
        # Stable limits of sin(t)/kappa and (1-cos(t))/kappa at kappa=0.
        xp = .5*kappa*l*l*np.sinc(t/(2*np.pi))**2
        zp = l*np.sinc(t/np.pi)
        cp, sp = np.cos(phi), np.sin(phi)
        return np.column_stack((xp*cp, xp*sp, zp)), t, cp, sp

    def backbone_xyz(self, s, kappa=0., phi=0.):
        """PCC centerline in the segment base frame, sampled at fiber coordinates s."""
        return self._backbone_geometry(s, kappa, phi)[0]

    def helix_xyz(self, s, kappa=0., phi=0.):
        """Bent surface in segment base frame, PCC; no dual-segment base transform."""
        s = np.asarray(s, dtype=float)
        backbone, t, cp, sp = self._backbone_geometry(s, kappa, phi)
        psi = s/self.lam
        vx, vy = self.config.radius_m*np.cos(psi), self.config.radius_m*np.sin(psi)
        v1x, v1y = vx*cp+vy*sp, -vx*sp+vy*cp
        return backbone + np.column_stack((v1x*np.cos(t)*cp-v1y*sp,
                                           v1x*np.cos(t)*sp+v1y*cp,
                                           -v1x*np.sin(t)))


def calibrate_force(force_n, peak_strain, fit_intercept=True):
    """Fit ONE segment/zone's |residual_peak| = slope*force + intercept."""
    f, y = np.asarray(force_n, float), np.abs(np.asarray(peak_strain, float))
    if f.ndim != 1 or f.shape != y.shape or len(f)<3 or not np.isfinite([f,y]).all() or np.any(f<0):
        raise ValueError('at least three finite matched force/strain samples required')
    x = np.column_stack((f, np.ones(len(f)))) if fit_intercept else f[:,None]
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('calibration needs distinct force levels')
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    if b[0] <= 0:
        raise ValueError('nonpositive sensitivity')
    return {'slope_strain_per_n': float(b[0]), 'intercept_strain': float(b[1]) if fit_intercept else 0.,
            'rmse_strain': float(np.sqrt(np.mean((y-x@b)**2)))}
