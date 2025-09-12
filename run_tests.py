#!/usr/bin/env python3
"""
Test runner for lemlem library.
Runs both fake API tests (always) and real API tests (if API key is available).
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("Lemlem Library Test Suite")
    print("=" * 60)
    
    # Change to the lemlem directory
    lemlem_dir = Path(__file__).parent
    os.chdir(lemlem_dir)
    print(f"Working directory: {os.getcwd()}")
    
    success_count = 0
    total_count = 0
    
    # Run fake API tests (always available)
    total_count += 1
    if run_command("python -m unittest tests.test_fake_api -v", "Fake API Tests"):
        success_count += 1
    
    # Check for OpenRouter API key
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if api_key:
        total_count += 1
        print(f"\n🔑 OpenRouter API key found (first 10 chars: {api_key[:10]}...)")
        if run_command("python -m unittest tests.test_real_openrouter -v", "Real API Tests (OpenRouter)"):
            success_count += 1
    else:
        print("\n⚠️  No OpenRouter API key found - skipping real API tests")
        print("   Set OPENROUTER_API_KEY to run real API tests")
        print("   Get a free key at: https://openrouter.ai/")
    
    # Run demo
    total_count += 1
    if run_command("python tests/demo_haiku.py", "Demo Script"):
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Tests passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())