"""
Helper script for managing releases.
Example usage in terminal:
python scripts/release.py patch  # or minor/major
"""

import subprocess
import sys
from typing import Literal


VersionBumpType = Literal["patch", "minor", "major"]


def run_command(command: str) -> None:
    """Run a shell command and exit on failure."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        sys.exit(1)


def bump_version(bump_type: VersionBumpType) -> str:
    """Bump the version using poetry."""
    result = subprocess.run(["poetry", "version", "-s"], capture_output=True, text=True, check=True)
    current_version = result.stdout.strip()

    run_command(f"poetry version {bump_type}")

    result = subprocess.run(["poetry", "version", "-s"], capture_output=True, text=True, check=True)
    new_version = result.stdout.strip()

    print(f"Bumped version from {current_version} to {new_version}")
    return new_version


def create_release(version_bump: VersionBumpType) -> None:
    """Create a new release."""

    # Bump version
    print(f"\nBumping {version_bump} version...")
    new_version = bump_version(version_bump)

    # Update files and commit
    print("\nCommitting version change...")
    run_command("git add pyproject.toml")
    run_command(f'git commit -m "chore: bump version to {new_version}"')

    # Create and push tag
    print("\nCreating and pushing tag...")
    run_command(f'git tag -a "v{new_version}" -m "Release version {new_version}"')
    run_command("git push origin main")
    run_command(f'git push origin "v{new_version}"')

    print(f"\nSuccessfully created release v{new_version}")


def validate_version_bump(bump_type: str) -> VersionBumpType:
    """Validate that the version bump type is valid."""
    if bump_type not in ("patch", "minor", "major"):
        print("Usage: release.py [patch|minor|major]")
        sys.exit(1)
    return bump_type  # type: ignore  # mypy knows this is safe due to the check above


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: release.py [patch|minor|major]")
        sys.exit(1)

    version_bump = validate_version_bump(sys.argv[1])
    create_release(version_bump)
