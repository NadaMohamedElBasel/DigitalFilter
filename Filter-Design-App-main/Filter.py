import numpy as np
from scipy import signal
import json
from pathlib import Path


class Filter:
    def __init__(self):
        self.zeros = []  # List of complex numbers
        self.poles = []  # List of complex numbers
        self.gain = 1.0
        self.all_pass_filters = []  # List of all-pass filters
        self.all_pass_zeros = []
        self.all_pass_poles = []

        # Create filters directory if it doesn't exist
        self.filters_dir = Path(__file__).parent / 'filters'
        self.filters_dir.mkdir(exist_ok=True)

        self.subscribers = []  # Subscribers should include callback functions for: Magnitude plot, Phase plot, and elements list.

    def subscribe(self, callback, instance):
        self.subscribers.append((callback, instance))

    def notify_subscribers(self, sender=None):
        self._normalize_gain()
        for callback, instance in self.subscribers:
            if sender is not instance:
                callback(self)

    def update_from_zplane(self, zeros, poles, all_pass_filters, sender):
        """Update filter coefficients from z-plane widget"""
        new_zeros = [complex(z.position.real, z.position.imag) for z in zeros]
        new_poles = [complex(p.position.real, p.position.imag) for p in poles]

        # Handle zero removal along with its conjugate
        for z in self.zeros:
            if z not in new_zeros:
                conjugate = complex(z.real, -z.imag)
                if conjugate in self.zeros:
                    self.zeros.remove(conjugate)

        # Handle pole removal along with its conjugate
        for p in self.poles:
            if p not in new_poles:
                conjugate = complex(p.real, -p.imag)
                if conjugate in self.poles:
                    self.poles.remove(conjugate)

        # Update zeros and poles
        self.zeros = new_zeros
        self.poles = new_poles

        self.all_pass_filters = all_pass_filters
        self.parse_all_pass_filters()

        # Notify subscribers
        self.notify_subscribers(sender)

    def parse_all_pass_filters(self):
        self.all_pass_zeros = []
        self.all_pass_poles = []
        for ap in self.all_pass_filters:
            a = ap["a"]
            angle = ap["theta"]
            zero = 1 / a * np.exp(1j * angle)
            pole = a * np.exp(1j * angle)
            self.all_pass_zeros.append(zero)
            self.all_pass_poles.append(pole)

    def update_from_element_list(self, zeros, poles, sender):
        self.zeros = zeros
        self.poles = poles
        self.notify_subscribers(sender)

    def update_all_pass_filters(self, all_pass_filters, sender):
        """Update the list of all-pass filters and notify subscribers"""
        self.all_pass_filters = all_pass_filters
        self.parse_all_pass_filters()
        self.notify_subscribers(sender)

    def _normalize_gain(self):
        """Normalize filter gain to 1 at DC (z = 1)"""
        if not self.zeros and not self.poles:
            self.gain = 1.0
            return

        # Calculate gain at z = 1 (DC)
        num = np.prod([1 - z for z in self.zeros]) if self.zeros else 1
        den = np.prod([1 - p for p in self.poles]) if self.poles else 1

        # Set gain to normalize DC response to 1
        self.gain = abs(num / den) if den != 0 else 1.0

    def get_transfer_function(self, caller=None):
        """Get filter coefficients in transfer function form"""
        if type(caller).__name__ in ['FilterCodeGenerator', 'FilterVisualizer']:
            # make sure filter is realizable
            if not self.is_realizable():
                raise ValueError("Filter must have a conjugate pair for each complex element to convert to transfer "
                                 "function form")

        return signal.zpk2tf(self.zeros + self.all_pass_zeros, self.poles + self.all_pass_poles, self.gain)

    def is_realizable(self):
        # Auto add conjugates for elements without a conjugate
        # If a complex element is present, it must have a conjugate
        for z in self.zeros:
            if z.imag != 0 and complex(z.real, -z.imag) not in self.zeros:
                return False
        for p in self.poles:
            if p.imag != 0 and complex(p.real, -p.imag) not in self.poles:
                return False
        for z in self.all_pass_zeros:
            if z.imag != 0 and complex(z.real, -z.imag) not in self.all_pass_zeros:
                return False
        for p in self.all_pass_poles:
            if p.imag != 0 and complex(p.real, -p.imag) not in self.all_pass_poles:
                return False
        return True

    def auto_realize_filter(self):
        # Auto add conjugates for elements without a conjugate
        # If a complex element is present, it must have a conjugate
        for z in self.zeros:
            if z.imag != 0 and complex(z.real, -z.imag) not in self.zeros:
                self.zeros.append(complex(z.real, -z.imag))
        for p in self.poles:
            if p.imag != 0 and complex(p.real, -p.imag) not in self.poles:
                self.poles.append(complex(p.real, -p.imag))
        for p in self.all_pass_poles:
            if p.imag != 0 and complex(p.real, -p.imag) not in self.all_pass_poles:
                self.all_pass_poles.append(complex(p.real, -p.imag))
                self.all_pass_zeros.append(complex(1 / p.real, -p.imag))
                self.all_pass_filters.append({"coefficient": 1 / p.real, "theta": -p.imag})
        self.notify_subscribers()

    def get_cascade_form(self):
        """Get filter coefficients in cascade form"""
        if not self.is_realizable():
            raise ValueError("Filter must have a conjugate pair for each complex element to convert to cascade form")
        return signal.zpk2sos(self.zeros, self.poles, self.gain)

    def get_frequency_response(self, num_points=1024):
        """Calculate frequency response"""
        w, h = signal.freqz(*self.get_transfer_function(), worN=num_points)

        # frequencies = w * self.sample_rate / (2 * np.pi)
        _epsilon = 1e-12
        magnitude_db = 20 * np.log10(np.abs(h) + _epsilon)
        phase_rad = np.angle(h)

        return w, magnitude_db, phase_rad

    def get_impulse_response(self, num_points=100):
        """Calculate impulse response"""
        b, a = self.get_transfer_function()
        return signal.lfilter(b, a, [1.0] + [0.0] * (num_points - 1))

    def save_to_file(self, filename):
        """Save filter to JSON file"""
        data = {
            'zeros': [(z.real, z.imag) for z in self.zeros],
            'poles': [(p.real, p.imag) for p in self.poles],
            'all_pass_filters': self.all_pass_filters,
            'gain': self.gain
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename):
        """Load filter from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.zeros = [complex(z[0], z[1]) for z in data['zeros']]
        self.poles = [complex(p[0], p[1]) for p in data['poles']]
        self.all_pass_filters = data['all_pass_filters']
        self.gain = data['gain']
        self.parse_all_pass_filters()
        self.notify_subscribers()
