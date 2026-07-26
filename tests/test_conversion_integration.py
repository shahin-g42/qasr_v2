import os
import tempfile
import unittest

from qasr.convert_weights import DEFAULT_QWEN, convert_components


@unittest.skipUnless(
    os.environ.get("QASR_RUN_COMPONENT_CONVERSION") == "1"
    and os.environ.get("QASR_ENCODER_CHECKPOINT"),
    "set QASR_RUN_COMPONENT_CONVERSION=1 and QASR_ENCODER_CHECKPOINT to load sources",
)
class ComponentConversionIntegrationTest(unittest.TestCase):
    def test_component_conversion_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            path = convert_components(
                encoder_name_or_path=os.environ["QASR_ENCODER_CHECKPOINT"],
                qwen_name_or_path=DEFAULT_QWEN,
                output_dir=output_dir,
                verify_reload=True,
            )

            self.assertEqual(str(path), output_dir)


if __name__ == "__main__":
    unittest.main()
