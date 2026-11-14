"""VS Code entry: synthetic demo or real CSV strain inversion."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from helical_fiber import Config, HelicalFiberModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV columns: s_m,strain (one frame)')
    parser.add_argument('--unit', choices=['strain', 'microstrain'], default='strain')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config/segment1.json')
    parser.add_argument('--method', choices=['tukey','ols'], default='tukey')
    parser.add_argument('--out', type=Path, default=Path('outputs/demo'))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    model = HelicalFiberModel(Config(**json.loads(args.config.read_text(encoding='utf-8'))))
    truth = None
    if args.input:
        data = np.genfromtxt(args.input, delimiter=',', names=True, encoding='utf-8-sig')
        s, y = np.atleast_1d(data['s_m']), np.atleast_1d(data['strain'])
        if args.unit == 'microstrain':
            y = y*1e-6
    else:
        s = np.linspace(0, model.config.fiber_length_m, 6395)
        truth = {'theta_deg': 30., 'phi_deg': 40., 'contacts': [
            {'s_m': 1.60, 'force_n': 2., 'width_m': .009},
            {'s_m': 2.65, 'force_n': 2.5, 'width_m': .012, 'sign': -1}]}
        synthetic = model.simulate(s, np.deg2rad(30)/.2, np.deg2rad(40), truth['contacts'])
        y = synthetic['strain']
    result = model.fit(s, y, method=args.method)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'config': asdict(model.config), 'synthetic_truth': truth,
              'summary': result['summary'], 'contacts': result['contacts']}
    (args.out/'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savetxt(args.out/'input.csv', np.column_stack((s,y)), delimiter=',', header='s_m,strain', comments='')
    np.savetxt(args.out/'decoupled.csv', np.column_stack((s,y,result['bending'],result['fitted'],result['residual'],result['weights'])),
               delimiter=',', header='s_m,strain,bending_strain,fitted_with_offset,residual_strain,tukey_weight', comments='')
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13,9), layout='constrained')
    ax = fig.add_subplot(311)
    ax.plot(s,y*1e6,lw=.6,label='Measured'); ax.plot(s,result['fitted']*1e6,lw=1,label='Fitted bending + offset')
    ax.set_ylabel('Strain (microstrain)'); ax.legend()
    ax = fig.add_subplot(312)
    ax.plot(s,result['residual']*1e6,lw=.8,label='Contact residual')
    ax.axhline(model.config.threshold_strain*1e6,c='gray',ls='--')
    ax.axhline(-model.config.threshold_strain*1e6,c='gray',ls='--')
    for contact in result['contacts']:
        ax.plot(contact['s_m'],contact['peak_strain']*1e6,'ro')
        ax.annotate(f"{contact['force_n']:.2f} N",(contact['s_m'],contact['peak_strain']*1e6))
    ax.set(xlabel='Fiber arc length (m)', ylabel='Residual (microstrain)'); ax.legend()
    ax = fig.add_subplot(313, projection='3d')
    xyz = model.helix_xyz(s,result['summary']['kappa_per_m'],result['summary']['phi_rad'] or 0.)
    ax.plot(*xyz.T,lw=.6)
    for contact in result['contacts']:
        ax.scatter(*contact['xyz_m'],c='r')
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.001))
    fig.savefig(args.out/'analysis.png',dpi=160)
    print(json.dumps({'summary':result['summary'],'contacts':result['contacts']},indent=2,ensure_ascii=False))
    print('Saved:', args.out.resolve())
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
