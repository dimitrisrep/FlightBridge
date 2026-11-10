"""Έλεγχος Streamlit με το πραγματικό model.pkl που παράγει το training."""
from pathlib import Path
import unittest
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_input_controls_and_real_predictions(self):
        project = Path(__file__).resolve().parents[1]
        samples = []
        original_predict = Pipeline.predict_proba

        def record_sample(pipe, sample, **kwargs):
            samples.append(sample.copy())
            return original_predict(pipe, sample, **kwargs)

        with patch.object(Pipeline, "predict_proba", record_sample):
            app = AppTest.from_file(str(project / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.radio[0].options), 4)
            app.selectbox[0].select("A3")
            app.number_input(key="delay_hours").set_value(0)
            app.number_input(key="delay_minutes").set_value(31).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(samples[-1].delay_minutes.iloc[0], 31)
            self.assertEqual(samples[-1].airline.iloc[0], "A3")
            self.assertEqual(set(samples[-1].columns), {
                "sc_type", "delay_minutes", "alternative_provided", "airline", "airline_type"
            })

            app.checkbox(key="over_one_day").check().run()
            self.assertTrue(all(widget.disabled for widget in app.number_input))
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(samples[-1].delay_minutes.iloc[0], 1441)

            app.checkbox(key="over_one_day").uncheck().run()
            self.assertTrue(all(not widget.disabled for widget in app.number_input))
            self.assertEqual(app.number_input(key="delay_minutes").value, 31)
            app.radio[0].set_value("Η πτήση μου ακυρώθηκε")
            app.radio[1].set_value("NO").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(samples[-1].delay_minutes.iloc[0], 0)

            app.selectbox[0].select("Άλλη — εταιρεία χαμηλού κόστους").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(samples[-1].airline.iloc[0], "UNKNOWN")
            self.assertEqual(samples[-1].airline_type.iloc[0], "LCC")


if __name__ == "__main__":
    unittest.main()
