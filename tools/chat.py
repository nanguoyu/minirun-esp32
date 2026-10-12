"""Send a message to the ESP32-S3 and watch its reply on the terminal and LCD."""
import argparse
import re
import subprocess
import sys
import time

from device import Device
from fetch_lfm import ROOT, verify

EOG = {2, 7}


def vocab(mode, args):
    helper = ROOT / 'build/reference'
    if not helper.exists():
        raise RuntimeError('Run bash tools/setup.sh to build the text helper.')
    result = subprocess.run([str(helper), str(ROOT / 'artifacts/model.gguf'), '-', mode] + args,
                            capture_output=True)
    if mode == '--tokenize' and result.returncode == 4:
        raise ValueError('This message is longer than the 1024-token input limit.')
    if result.returncode:
        raise RuntimeError('The text helper failed: ' + result.stderr.decode(errors='replace')[-400:])
    return result.stdout.decode('utf-8', errors='replace')


def tokenize(prompt):
    text = '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    line = next(line for line in vocab('--tokenize', [text]).splitlines() if line.startswith('TOKENS'))
    return list(map(int, line.split()[1:]))


def validate_budget(prompt_count, new_tokens, capacity):
    if new_tokens < 1:
        raise ValueError('--new-tokens must be at least 1.')
    if prompt_count + new_tokens - 1 > capacity:
        raise ValueError(f'The message uses {prompt_count} tokens; with an output limit of {new_tokens}, '
                         f'it exceeds this board\'s {capacity}-token context. Shorten the message or output limit.')


def show(device, text):
    renderer = ROOT / 'build/render_text'
    if renderer.exists():
        frame = subprocess.run([str(renderer), text], capture_output=True, check=True).stdout
        if len(frame) != 17280:
            raise RuntimeError('Unexpected LCD frame size.')
        device.command('FRAME')
        device.until('FRAME_READY')
        if device.s.write(frame) != len(frame):
            raise IOError('Incomplete LCD transfer.')
        device.until('FRAME_OK')
    else:
        # The firmware includes an ASCII font for hosts without Cocoa.
        payload = text.encode('ascii', errors='replace')[-512:]
        device.command(f'TEXT {len(payload)}')
        device.until('TEXT_READY')
        if device.s.write(payload) != len(payload):
            raise IOError('Incomplete LCD text transfer.')
        device.until('TEXT_OK')


class Progress:
    def __init__(self):
        self.stage = ''
        self.terminal = sys.stderr.isatty()

    def set(self, stage):
        self.stage = stage
        if self.terminal:
            print('\r\033[K' + stage, end='', file=sys.stderr, flush=True)
        else:
            print(stage, file=sys.stderr, flush=True)

    def line(self, line):
        match = re.match(r'LAYER (\d+)', line)
        if self.terminal and match:
            print(f'\r\033[K{self.stage} · layer {int(match[1]) + 1}/14',
                  end='', file=sys.stderr, flush=True)

    def clear(self):
        if self.terminal:
            print('\r\033[K', end='', file=sys.stderr, flush=True)


def run(args):
    model = ROOT / 'artifacts/model.gguf'
    if not model.exists():
        raise ValueError('Download the model first: python tools/fetch_lfm.py')
    verify(model)
    ids = tokenize(args.prompt)
    progress = Progress()
    device = Device(args.port, verbose=args.verbose, on_line=progress.line)
    generated = []
    completed = False
    file_cache = False
    started = time.monotonic()
    try:
        device.command('INFO')
        info = device.until('READY LFM230')
        capacity = int(re.search(r' context=(\d+)', info)[1])
        validate_budget(len(ids), args.new_tokens, capacity)
        file_cache = 'kv_backend=FILE' in info
        print(f'You: {args.prompt}\n', flush=True)
        show(device, 'You: ' + args.prompt + '\n\nThinking...')
        device.command('MODE ' + args.mode)
        device.until('MODE_OK')
        reuse = False
        if not args.reverify:
            device.command('RESET')
            answer = device.until('RESET_')
            reuse = answer.startswith('RESET_OK')
            if not reuse and answer != 'RESET_NEEDS_OPEN':
                raise RuntimeError(answer)
        if not reuse:
            progress.set('Opening the model on the SD card...')
            device.command('PHASE VERIFYING')
            device.until('PHASE_OK')
            device.command('OPEN')
            device.until('OPENED', 600)
        for base in range(0, len(ids), args.batch_size):
            chunk = ids[base:base + args.batch_size]
            progress.set(f'Reading your message: {base + 1}-{base + len(chunk)}/{len(ids)}')
            device.command(f'PHASE PREFILL {base + 1}-{base + len(chunk)}')
            device.until('PHASE_OK')
            head = int(base + len(chunk) == len(ids))
            device.command('PREFILL ' + str(head) + ' ' + ' '.join(map(str, chunk)))
            result = device.until('TOKEN ', 900)
        progress.clear()
        print('Minirun: ', end='', flush=True)
        previous = ''
        reason = 'output limit reached'
        for step in range(args.new_tokens):
            token = int(re.search(r'argmax=(\d+)', result)[1])
            if token in EOG:
                reason = 'finished'
                break
            generated.append(token)
            answer = vocab('--decode', list(map(str, generated)))
            # Decode the whole prefix: a Unicode character can span several tokens.
            stable = answer.rstrip('\ufffd')
            if stable.startswith(previous):
                print(stable[len(previous):], end='', flush=True)
                previous = stable
            show(device, 'You: ' + args.prompt + '\n\n' + answer)
            if step == args.new_tokens - 1:
                break
            progress.set(f'Writing the reply: token {step + 2}/{args.new_tokens}')
            device.command(f'PHASE REPLY {step + 2}/{args.new_tokens}')
            device.until('PHASE_OK')
            device.command(f'STEP {token} 1')
            result = device.until('TOKEN ', 900)
            progress.clear()
        print('\n', flush=True)
        if file_cache:
            device.command('KVCLOSE')
            device.until('KV_CLOSED')
        device.command('PHASE DONE' if reason == 'finished' else 'PHASE OUTPUT LIMIT')
        device.until('PHASE_OK')
        completed = True
        progress.set(f'{len(generated)} tokens · {time.monotonic() - started:.0f}s · {reason}')
    except KeyboardInterrupt:
        device.s.write(b'!')
        progress.clear()
        print('\nStopping the board...', file=sys.stderr)
        try:
            device.until('ERROR', 30)
        except RuntimeError:
            if file_cache:
                device.command('KVCLOSE')
                device.until('KV_CLOSED', 10)
        except TimeoutError:
            pass
        raise
    finally:
        progress.clear()
        if completed and not sys.stderr.isatty():
            sys.stderr.flush()
        device.close()
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='USB port; detected automatically for one connected ESP32-S3')
    parser.add_argument('--prompt', help='Message to send; asks in the terminal when omitted')
    parser.add_argument('--new-tokens', type=int, default=32, help='Maximum reply length (default: 32)')
    parser.add_argument('--batch-size', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--reverify', action='store_true', help='Check the whole model file again')
    parser.add_argument('--mode', choices=['PIPELINE', 'SERIAL'], default='PIPELINE')
    parser.add_argument('--verbose', action='store_true', help='Print firmware diagnostics')
    args = parser.parse_args()
    try:
        if args.prompt is None:
            args.prompt = input('Message: ')
        if not args.prompt.strip():
            parser.error('Enter a message to send.')
        run(args)
    except (KeyboardInterrupt, EOFError):
        print('\nStopped.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        parser.exit(1, str(error) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
