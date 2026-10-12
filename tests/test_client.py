import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import chat
import device


class ClientTests(unittest.TestCase):
    def test_context_includes_message_and_generated_positions(self):
        chat.validate_budget(1000, 25, 1024)
        for args in [(1000, 26, 1024), (1025, 1, 1024), (2, 0, 1024)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                chat.validate_budget(*args)

    def test_port_selection_never_chooses_between_two_boards(self):
        board = SimpleNamespace(vid=0x303A, pid=0x1001, device='/dev/esp-test')
        other = SimpleNamespace(vid=0x1234, pid=0x1001, device='/dev/other')
        with patch.dict('os.environ', {}, clear=True), patch.object(device.list_ports, 'comports') as ports:
            ports.return_value = [other]
            with self.assertRaises(ValueError):
                device.resolve_port()
            ports.return_value = [other, board]
            self.assertEqual(device.resolve_port(), board.device)
            ports.return_value = [board, board]
            with self.assertRaises(ValueError):
                device.resolve_port()
            self.assertEqual(device.resolve_port('/dev/chosen'), '/dev/chosen')

    def test_template_has_one_user_turn_and_open_assistant_turn(self):
        with patch.object(chat, 'vocab', return_value='TOKENS 1 6 7\n') as vocab:
            self.assertEqual(chat.tokenize('Hello!'), [1, 6, 7])
            self.assertEqual(vocab.call_args.args,
                             ('--tokenize', ['<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n']))

    def test_both_model_end_markers_finish_without_an_extra_step(self):
        for marker in (2, 7):
            commands = []
            class Board:
                def __init__(self, *args, **kwargs): pass
                def command(self, command): commands.append(command)
                def close(self): commands.append('CLOSED')
                def until(self, prefix, *args):
                    return {'READY LFM230': 'READY LFM230 context=1024 kv_backend=FILE',
                            'RESET_': 'RESET_OK',
                            'TOKEN ': f'TOKEN position=3 argmax={marker}'} .get(prefix, prefix)
            args = SimpleNamespace(port=None, verbose=False, prompt='Hello!', new_tokens=8,
                                   batch_size=8, reverify=False, mode='PIPELINE')
            with self.subTest(marker=marker), patch.object(chat.Path if hasattr(chat, 'Path') else type(chat.ROOT), 'exists', return_value=True), patch.object(chat, 'verify'), patch.object(chat, 'tokenize', return_value=[1, 6, 7]), patch.object(chat, 'show'), patch.object(chat, 'Device', Board):
                self.assertEqual(chat.run(args), [])
            self.assertFalse(any(command.startswith('STEP ') for command in commands))
            self.assertIn('KVCLOSE', commands)
            self.assertEqual(commands[-1], 'CLOSED')


if __name__ == '__main__':
    unittest.main()
