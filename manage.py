#!/usr/bin/env python
import os
import sys

# Виправлення проблем з кодуванням Windows
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def main() -> None:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()


