#!/localdata0/Guests/flynne03/Python/conda/miniconda3/envs/py_ms/bin/python

import argparse
from dashapp.qc import app as qc_app
from dashapp.fileIO import app as file_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Mass Spec QA/QC Webapp")

    # parser.add_argument('--debug',    action='store_true',  dest='debug_mode', help="Run in debug mode.")
    # parser.set_defaults(debug_mode=False)
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug_mode",
        help="Do not run in debug mode.",
    )
    parser.set_defaults(debug_mode=True)
    parser.add_argument(
        "--port", type=int, help="port on which to run dash server", default=8050
    )
    args = parser.parse_args()

    # qc_app.run_server(host='0.0.0.0', port=args.port, debug=args.debug_mode)
    file_app.run_server(host='0.0.0.0', port=args.port, debug=args.debug_mode)

