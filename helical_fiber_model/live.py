"""Reconstruct segment-local shape and contact candidates from ODiSI measurements."""
import json
import logging
from pathlib import Path
import time

import numpy as np
from odisi_tcp import ODISIReceiver
from odisi_tcp.receiver import fields
from odisi_tcp.stream import emit_result, parser, run_processor

from helical_fiber import Config, HelicalFiberModel


def sensing_coordinates(metadata, channel, args, config, point_count):
    if args.fiber_length is not None:
        if args.stop_index is not None and args.stop_index > point_count:
            raise ValueError('measurement ends before --stop-index')
        crop = slice(args.start_index, args.stop_index)
        count = len(range(point_count)[crop])
        return crop, np.linspace(0, args.fiber_length, count)

    for sensor in metadata.get('sensors', []):
        if sensor.get('channel') != channel:
            continue
        for segment in sensor.get('segments', []):
            if segment.get('segment name') != 'All Gages':
                continue
            start, count = segment['index'], segment['size']
            if start < 0 or count < 10 or start + count > point_count:
                raise ValueError('All Gages lies outside the measurement or has too few points')
            pitch = float(sensor['gage pitch (mm)']) * 1e-3
            span = pitch * (count - 1)
            if not np.isfinite(pitch) or pitch <= 0:
                raise ValueError('invalid gage pitch in ODiSI metadata')
            if args.sensor_start_m is not None:
                origin = float(segment['location (mm)']) * 1e-3
                if not np.isfinite(origin):
                    raise ValueError('invalid All Gages origin in ODiSI metadata')
                local_s = origin + np.arange(count) * pitch - args.sensor_start_m
                indices = np.flatnonzero((local_s >= -1e-10) & (local_s <= config.fiber_length_m + 1e-10))
                if len(indices) < 10:
                    raise ValueError('fewer than 10 measured points inside the configured sensing span')
                first, stop = indices[0], indices[-1] + 1
                return slice(int(start + first), int(start + stop)), np.clip(local_s[first:stop], 0, config.fiber_length_m)
            full_span = np.isclose(span, config.fiber_length_m, rtol=0, atol=1e-10)
            if not full_span and not (args.assume_span_at_base and span < config.fiber_length_m):
                raise ValueError(f'All Gages covers {span * 1000:g} mm, '
                                 f'while model fiber_length_m is {config.fiber_length_m * 1000:g} mm')
            return slice(start, start + count), np.arange(count) * pitch
    raise ValueError('waiting for ODiSI All Gages metadata; manual input requires --fiber-length')


class LivePlot:
    radius_m = .0175

    def __init__(self, models):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.plt = plt
        self.models = models
        angles = np.arange(16) * (2 * np.pi / 16)
        self.ring_xy = self.radius_m * np.column_stack((np.cos(angles), np.sin(angles)))
        plt.ion()
        self.figure, axes = plt.subplots(1, len(models), squeeze=False,
                                       subplot_kw={'projection': '3d'}, figsize=(6 * len(models), 5))
        self.figure.subplots_adjust(left=.08, right=.94, bottom=.10, top=.84, wspace=.18)
        self.artists = {}
        for (channel, model), ax in zip(models.items(), axes[0]):
            reach = model.config.backbone_length_m + self.radius_m
            ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=f'Channel {channel}',
                   xlim=(-reach, reach), ylim=(-reach, reach), zlim=(-reach, reach))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-60)
            backbone = Poly3DCollection([], facecolors='#397ca6', edgecolors='none', alpha=.72)
            ax.add_collection3d(backbone)
            contacts, = ax.plot([], [], [], linestyle='none', marker='o', color='tab:red',
                                markeredgecolor='white', markersize=8, zorder=10)
            status = ax.text2D(.02, .96, 'Waiting for measurement', transform=ax.transAxes, va='top')
            self.artists[channel] = backbone, contacts, status
        self.pending = {}
        self.next_draw = 0.
        plt.show(block=False)

    def update(self, *, force=False):
        if not self.plt.fignum_exists(self.figure.number):
            return False
        now = time.monotonic()
        if self.pending and (force or now >= self.next_draw):
            for channel, result in self.pending.items():
                backbone, markers, status = self.artists[channel]
                if 'error' in result:
                    backbone.set_verts([])
                    markers.set_data_3d([], [], [])
                    status.set_text(result['error'])
                    continue
                summary = result['summary']
                model = self.models[channel]
                center = np.asarray(result['shape']['backbone_xyz_m'])
                kappa = summary['kappa_per_m']
                phi = (summary['phi_rad'] or 0.) + model.config.geometry_phi_offset_rad
                turn = kappa * np.linspace(0, model.config.backbone_length_m, len(center))
                normal = np.column_stack((np.cos(turn) * np.cos(phi),
                                          np.cos(turn) * np.sin(phi), -np.sin(turn)))
                binormal = np.array([-np.sin(phi), np.cos(phi), 0.])
                rings = (center[:, None, :] + normal[:, None, :] * self.ring_xy[None, :, :1]
                         + binormal * self.ring_xy[None, :, 1:])
                adjacent = np.roll(rings, -1, axis=1)
                sides = np.stack((rings[:-1], adjacent[:-1], adjacent[1:], rings[1:]), axis=2)
                backbone.set_verts([*sides.reshape(-1, 4, 3), rings[0][::-1], rings[-1]])

                contacts = result['contacts']
                if contacts:
                    contact_center = model.backbone_xyz([c['s_m'] for c in contacts], kappa,
                                                        summary['phi_rad'] or 0.)
                    radial = np.asarray([c['xyz_m'] for c in contacts]) - contact_center
                    surface = contact_center + radial * (self.radius_m / model.config.radius_m)
                    markers.set_data_3d(*surface.T)
                else:
                    markers.set_data_3d([], [], [])
                label = f"Bend {summary['theta_deg']:.1f}° | {len(contacts)} contact candidates"
                if summary['phi_deg'] is not None:
                    label += f"\nBending-plane azimuth {summary['phi_deg']:.1f}°"
                tip_x, tip_y, _ = result['shape']['backbone_xyz_m'][-1]
                label += f'\nTip X {tip_x * 1000:+.1f} mm | Y {tip_y * 1000:+.1f} mm'
                if result['shape']['extrapolated']:
                    label += '\nExtrapolated from a partial window'
                status.set_text(label)
            self.pending.clear()
            self.figure.canvas.draw_idle()
            self.next_draw = now + .2
        self.figure.canvas.flush_events()
        return self.plt.fignum_exists(self.figure.number)


def main():
    p = parser(__doc__, require_fiber_length=False)
    p.add_argument('--config', type=Path, help='geometry and bending calibration JSON for --channel')
    p.add_argument('--segment', nargs=2, action='append', metavar=('CHANNEL', 'CONFIG'),
                   help='repeat for each independently reconstructed channel')
    p.add_argument('--replay', type=Path, help='process a receive.py messages.jsonl file without connecting')
    p.add_argument('--plot', action='store_true', help='show segment-local 3D centerlines and contact candidates')
    p.add_argument('--assume-span-at-base', action='store_true',
                   help='treat a shorter All Gages window as starting at the base; extrapolate PCC shape')
    p.add_argument('--sensor-start-m', type=float,
                   help='sensing fiber start in ODiSI coordinates (m), using All Gages metadata')
    p.add_argument('--baseline', type=Path, help='single-channel CSV with s_m and strain_microstrain columns')
    p.add_argument('--force-calibrated', action='store_true', help='include force estimates using the supplied calibrations')
    args = p.parse_args()
    if args.config and args.segment:
        p.error('use either --config or --segment')
    if args.sensor_start_m is not None and (
            not np.isfinite(args.sensor_start_m) or args.sensor_start_m < 0
            or args.fiber_length is not None or args.assume_span_at_base):
        p.error('--sensor-start-m requires a finite nonnegative coordinate and metadata-based cropping')
    if args.fiber_length is None and (args.start_index or args.stop_index is not None):
        p.error('manual cropping requires --fiber-length')
    if args.replay and args.duration:
        p.error('--duration applies to live TCP only')
    if (not np.isfinite(args.duration) or args.duration < 0 or args.start_index < 0
            or (args.stop_index is not None and args.stop_index <= args.start_index)):
        p.error('invalid duration or crop')
    if args.replay and not args.replay.is_file():
        p.error(f'replay file not found: {args.replay}')
    try:
        models = {}
        default_config = Path(__file__).parent / 'config' / 'segment1.json'
        for channel, path in args.segment or [(args.channel, args.config or default_config)]:
            channel = int(channel)
            if channel < 1 or channel in models:
                raise ValueError('channel numbers must be positive and unique')
            config = Config(**json.loads(Path(path).read_text(encoding='utf-8')))
            if config.live_fit_window_m is not None and args.sensor_start_m is None:
                raise ValueError('live_fit_window_m requires --sensor-start-m')
            if args.fiber_length is not None and not np.isclose(args.fiber_length, config.fiber_length_m):
                raise ValueError('--fiber-length must match each model config')
            models[channel] = HelicalFiberModel(config)
    except (OSError, ValueError, TypeError) as exc:
        p.error(str(exc))

    baseline = None
    if args.baseline:
        if len(models) != 1 or args.sensor_start_m is None:
            p.error('--baseline requires one channel and --sensor-start-m')
        try:
            samples = np.genfromtxt(args.baseline, delimiter=',', names=True, ndmin=1)
            baseline_s, baseline = samples['s_m'], samples['strain_microstrain'] * 1e-6
            if (len(baseline_s) < 10 or not np.all(np.isfinite(baseline_s))
                    or np.any(np.diff(baseline_s) <= 0)):
                raise ValueError('baseline requires at least 10 ordered finite coordinates')
        except (OSError, ValueError) as exc:
            p.error(str(exc))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    metadata = {}
    connection = None
    reported_errors = {}
    plot = LivePlot(models) if args.plot else None

    def receive_message(event):
        nonlocal metadata, connection
        if event['connection_number'] != connection:
            connection = event['connection_number']
            metadata = {}
            if args.replay:
                reported_errors.clear()
                if plot:
                    plot.pending.update({channel: {'error': 'Waiting for measurement'} for channel in models})
        message = event['message']
        if str(fields(message).get('messagetype', '')).strip().lower() == 'metadata':
            if message.get('sensors') or not metadata:
                metadata = message

    def process(measurement):
        channel = measurement.channel
        model = models[channel]
        try:
            crop, s = sensing_coordinates(metadata, channel, args, model.config, len(measurement.data))
            y = np.asarray(measurement.data[crop], dtype=float)
            if args.unit == 'microstrain':
                y *= 1e-6
            if baseline is not None:
                first = int(np.searchsorted(s, baseline_s[0] - 1e-8))
                stop = first + len(baseline_s)
                if stop > len(s) or not np.allclose(s[first:stop], baseline_s, atol=1e-8, rtol=0):
                    raise ValueError('live sample grid does not match the calibration baseline')
                s, y = s[first:stop], y[first:stop] - baseline
                crop = slice(crop.start + first, crop.start + stop)
            if model.config.live_fit_window_m is not None:
                lower, upper = model.config.live_fit_window_m
                first = int(np.searchsorted(s, lower - 1e-10))
                stop = int(np.searchsorted(s, upper + 1e-10, side='right'))
                s, y = s[first:stop], y[first:stop]
                crop = slice(crop.start + first, crop.start + stop)
            valid_count = np.count_nonzero(np.isfinite(y))
            if valid_count < 10:
                raise ValueError(f'only {valid_count}/{len(y)} valid strain samples; at least 10 required')
            fitted = model.fit(s, y)
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
            if plot:
                plot.pending[channel] = {'error': error}
            if reported_errors.get(channel) != error:
                logging.warning('Channel %d: %s', channel, error)
                reported_errors[channel] = error
                return {'error': error}
            return None

        reported_errors.pop(channel, None)
        summary, contacts = fitted['summary'], fitted['contacts']
        extrapolated = s[-1] < model.config.fiber_length_m - 1e-10 or s[0] > 1e-10
        if args.sensor_start_m is not None and extrapolated:
            summary['warnings'].append('Full-segment shape extends beyond the sampled sensing coordinates')
        elif extrapolated:
            summary['warnings'].append('Partial window assumed to start at fiber s=0; full-segment shape is extrapolated')
        if s[-1] - s[0] < 2 * np.pi * model.lam / model.config.frequency_modifier:
            summary['warnings'].append('Measurement covers less than one carrier period')
        for contact in contacts:
            contact['odisi_index'] = crop.start + contact['sample_index']
            if not args.force_calibrated:
                del contact['force_n'], contact['calibration_extrapolated']
        shape_s = np.linspace(0, model.config.fiber_length_m, 64)
        xyz = model.backbone_xyz(shape_s, summary['kappa_per_m'], summary['phi_rad'] or 0.)
        result = dict(summary=summary, contacts=contacts, shape={
            'frame': 'segment_base', 'backbone_xyz_m': xyz.tolist(),
            'observed_fiber_span_m': [float(s[0]), float(s[-1])], 'extrapolated': bool(extrapolated),
        })
        if plot:
            plot.pending[channel] = result
        return result

    try:
        if args.replay:
            receiver = ODISIReceiver(channels=models)
            with args.replay.open(encoding='utf-8') as log:
                for line in log:
                    event = json.loads(line)
                    receive_message(event)
                    measurement = receiver.handle_message(event['message'])
                    if measurement is not None:
                        emit_result(measurement, process(measurement), received_at=event['received_at'])
                    if plot and not plot.update():
                        break
            if plot and plot.update(force=True):
                plot.plt.ioff()
                plot.plt.show()
        else:
            run_processor(args, process, idle=plot.update if plot else None,
                          on_message=receive_message, channels=models,
                          latest_interval=.2 if plot else None)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError) as exc:
        p.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
