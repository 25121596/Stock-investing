import io
import unittest
import numpy as np
import pandas as pd


class FeatureCacheTests(unittest.TestCase):
    def test_csv_round_trip_preserves_boundary_floats(self):
        original = pd.DataFrame({'feature': [.1 + .2, np.nextafter(.005, 0),
                                             -.003024831684062287, 0.000000000145769]})
        stream = io.StringIO()
        original.to_csv(stream, index=False)
        stream.seek(0)
        restored = pd.read_csv(stream, float_precision='round_trip')
        np.testing.assert_array_equal(original.to_numpy(), restored.to_numpy())


if __name__ == '__main__':
    unittest.main()
