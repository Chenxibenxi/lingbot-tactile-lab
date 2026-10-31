import unittest
import json
from pathlib import Path
import numpy as np
from tactile_rpc import attach_tactile, encode_tactile_window, decode_tactile, FIELDS


def window(step=25, latest=25):
    return {"schema":"phizju_tactile_window_v1", "anchor_physics_step":step,
            "dt_s":.004,"features":[[[1.,2.,.5] for _ in range(4)] for _ in range(5)],
            "sample_fraction_mask":[[1.]*4 for _ in range(5)],
            "quality":{"window_valid":True,"last_source_step":latest,
                       "force_age_ms":(step-latest)*4.}}


class TactileRPCContractTests(unittest.TestCase):
    def test_versioned_candidate_config_matches_wire_schema(self):
        p=Path(__file__).resolve().parents[1]/'configs/tactile_rpc_v1.json'
        config=json.loads(p.read_text())
        self.assertEqual(tuple(config['optional_payload_keys']),FIELDS)
        self.assertEqual(config['candidate_max_age_ms'],200)
        self.assertEqual(config['status'],'cpu_contract_tested_not_wired_into_policy')

    def test_disabled_is_original_dictionary_and_no_extra_keys(self):
        original={"observation.state":np.ones(14),"task":"lift the pot"}
        self.assertIs(attach_tactile(original,window(),enabled=False),original)
        self.assertIsNone(decode_tactile(original))

    def test_causal_limited_payload_no_object_names_or_future(self):
        obs={"observation.state":np.ones(14),"task":"lift the pot"}
        result=attach_tactile(obs,{**window(),"objects":["privileged_cube"],
                                    "target_future_force":1e9},enabled=True)
        self.assertEqual(set(result)-set(obs),set(FIELDS))
        self.assertEqual(result[FIELDS[0]].shape,(5,4,3))
        self.assertTrue(decode_tactile(result)["valid"][0])
        self.assertNotIn('privileged_cube',repr(result))

    def test_msgpack_numpy_wire_roundtrip_preserves_shapes(self):
        import msgpack
        import msgpack_numpy
        payload=attach_tactile({'observation.state':np.ones(14)},window(),enabled=True)
        restored=msgpack.unpackb(msgpack.packb(payload,default=msgpack_numpy.encode),
                                 object_hook=msgpack_numpy.decode)
        parsed=decode_tactile(restored)
        self.assertEqual(parsed['history'].shape,(5,4,3))
        self.assertEqual(parsed['mask'].shape,(5,4))
        self.assertTrue(parsed['valid'][0])

    def test_actual_robotwin_websocket_numpy_packer_roundtrip(self):
        import importlib.util
        from pathlib import Path
        source=Path('/mnt/relaydata/lingbot/RoboTwin/script/deploy/msgpack_numpy.py')
        spec=importlib.util.spec_from_file_location('robotwin_msgpack_numpy_contract',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        payload=attach_tactile({'observation.state':np.ones(14)},window(),enabled=True)
        restored=module.unpackb(module.Packer().pack(payload))
        self.assertEqual(restored[FIELDS[0]].shape,(5,4,3))
        self.assertTrue(decode_tactile(restored)['valid'][0])

    def test_bad_future_or_inconsistent_age_rejected(self):
        with self.assertRaisesRegex(ValueError,'future'):
            encode_tactile_window(window(step=24,latest=25))
        bad=window();bad['quality']['force_age_ms']=100
        with self.assertRaisesRegex(ValueError,'age inconsistent'):
            encode_tactile_window(bad)

    def test_sensor_missing_and_stale_fallback_and_partial_payload_rejected(self):
        for candidate in (None,{**window(),"quality":{
                "window_valid":False,"last_source_step":None,"force_age_ms":None}}):
            parsed=decode_tactile(encode_tactile_window(candidate))
            self.assertFalse(parsed['valid'][0]);self.assertFalse(np.any(parsed['mask']))
        old=window(step=100,latest=25)
        parsed=decode_tactile(encode_tactile_window(old))
        self.assertFalse(parsed['valid'][0])
        message=encode_tactile_window(window())
        message.pop(FIELDS[2])
        with self.assertRaisesRegex(ValueError,'partial'):
            decode_tactile(message)

    def test_server_rejects_untrusted_invalid_mask_and_future_flags(self):
        msg=encode_tactile_window(window())
        msg[FIELDS[1]][0,0]=2.
        with self.assertRaisesRegex(ValueError,'bounds'):
            decode_tactile(msg)
        msg=encode_tactile_window(window())
        msg[FIELDS[3]]=np.asarray([False],np.bool_)
        with self.assertRaisesRegex(ValueError,'zero-masked'):
            decode_tactile(msg)


if __name__=='__main__':unittest.main()
