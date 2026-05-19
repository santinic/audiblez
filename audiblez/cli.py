# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import voices, available_voices_str


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              available_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-d', '--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device: auto (default), cpu, cuda (NVIDIA), or mps (Apple Metal)')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help='Model precision: fp32 (default), bf16 (faster on MPS/CUDA), fp16 (faster but may crash on MPS)')
    parser.add_argument('-c', '--cuda', default=False, action='store_true',
                        help='Deprecated; equivalent to --device cuda')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    preference = args.device
    if args.cuda and args.device == 'auto':
        print('Note: -c/--cuda is deprecated; use --device cuda')
        preference = 'cuda'

    from audiblez.core import main, set_device
    device = set_device(preference)
    print(f'Resolved device: {device}')
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output,
         device=device, precision=args.precision)


if __name__ == '__main__':
    cli_main()
