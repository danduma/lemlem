#!/usr/bin/env python3
import json
import sys


def main():
    payload = {
        "argv": sys.argv[1:],
        "stdin": sys.stdin.read(),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
