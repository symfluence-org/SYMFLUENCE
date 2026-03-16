# SYMFLUENCE - npm Package

Pre-compiled hydrological modeling tools for SYMFLUENCE framework.

## What's Included

This package provides pre-built binaries for:

- **SUMMA** - Structure for Unifying Multiple Modeling Alternatives
- **mizuRoute** - Multi-scale routing model
- **FUSE** - Framework for Understanding Structural Errors
- **NGEN** - NOAA Next Generation Water Resources Modeling Framework
- **TauDEM** - Terrain Analysis Using Digital Elevation Models

## Installation

### Global Installation (Recommended)

```bash
npm install -g symfluence
```

This will:
1. Download platform-specific pre-compiled binaries (~50-100 MB)
2. Extract them to your global npm directory
3. Make the `symfluence` command available

### Local Installation

```bash
npm install symfluence
```

## Supported Platforms

- **Linux**: x86_64 (Ubuntu 22.04+, RHEL 9+, Debian 12+)
- **macOS**: ARM64 (Apple Silicon M1/M2/M3, macOS 12+)

## System Requirements

### Linux

- **OS**: Ubuntu 22.04+, RHEL 9+, or Debian 12+
- **glibc**: ≥ 2.35
- **Libraries** (must be installed):
  ```bash
  sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103 libgdal32
  ```

### macOS

- **OS**: macOS 12 (Monterey) or later
- **Architecture**: Apple Silicon (ARM64)
- **Libraries** (install via Homebrew):
  ```bash
  brew install netcdf netcdf-fortran hdf5 gdal
  ```

For detailed requirements, see [SYSTEM_REQUIREMENTS.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md).

## Usage

### Check Installation

```bash
symfluence info
```

This shows:
- Installed version
- Platform information
- Available tools
- Build metadata
- Binary directory path

### Use Tools Directly

#### Option 1: Add to PATH

```bash
# Bash/Zsh
export PATH="$(npm root -g)/symfluence/dist/bin:$PATH"

# Fish
set -x PATH (npm root -g)/symfluence/dist/bin $PATH
```

Then run tools directly:
```bash
summa --version
mizuroute --help
ngen --version
```

#### Option 2: Use Full Path

```bash
$(npm root -g)/symfluence/dist/bin/summa --version
```

#### Option 3: Use with SYMFLUENCE Python Package

```bash
# Install Python package
pip install symfluence

# Configure to use npm-installed binaries
export SYMFLUENCE_DATA="$(npm root -g)/symfluence/dist"
```

### Get Binary Directory

```bash
symfluence path
```

## Commands

```bash
symfluence info       # Show installation info and available tools
symfluence version    # Show version
symfluence path       # Show binary directory path
symfluence help       # Show help
```

## Troubleshooting

### Installation Fails

1. **Check platform support**:
   ```bash
   node -e "console.log(process.platform, process.arch)"
   ```
   Must be `linux x64` or `darwin arm64`

2. **Check internet connection**: Downloads from GitHub Releases

3. **Verify release exists**:
   https://github.com/symfluence-org/SYMFLUENCE/releases

4. **Try manual installation**: See repository README

### "libnetcdf.so.19: not found" (Linux)

Install required libraries:
```bash
sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103
```

### "dyld: Library not loaded" (macOS)

Install required libraries:
```bash
brew install netcdf netcdf-fortran hdf5
```

### "version `GLIBC_2.35' not found" (Linux)

Your system has an older glibc. Options:
- Upgrade to Ubuntu 22.04+ / RHEL 9+ / Debian 12+
- Build from source (see repository docs)
- Use Docker (see repository docs)

## Development

### Local Testing

```bash
# In the npm/ directory
npm install .          # Test installation
node install.js        # Test download manually
./bin/symfluence info  # Test CLI
```

### Publishing

```bash
# Update version in package.json to match release tag
npm publish
```

## Documentation

- **Repository**: https://github.com/symfluence-org/SYMFLUENCE
- **System Requirements**: [docs/SYSTEM_REQUIREMENTS.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md)
- **Dynamic Linking Strategy**: [docs/DYNAMIC_LINKING_STRATEGY.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/DYNAMIC_LINKING_STRATEGY.md)
- **Issues**: https://github.com/symfluence-org/SYMFLUENCE/issues

## License

GPL-3.0 - See repository for details.

## Contributing

This package provides pre-built binaries only. For contributing to the tools themselves or the Python framework, see the main repository.

## Credits

- **SUMMA**: Martyn Clark and NCAR
- **mizuRoute**: Naoki Mizukami and NCAR
- **FUSE**: Martyn Clark
- **NGEN**: NOAA-OWP
- **TauDEM**: David Tarboton, Utah State University

SYMFLUENCE framework developed by Darri Eythorsson.
