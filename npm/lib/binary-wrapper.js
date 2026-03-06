/**
 * Generic CLI wrapper for symfluence bundled binaries.
 *
 * Each tool shim (bin/ngen, bin/summa, etc.) requires this module
 * and it resolves the native binary from dist/bin/<toolName>.
 *
 * For ngen: automatically patches library_file paths in the realization
 * config so users don't need to manually set paths to bundled BMI libs.
 */

const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * BMI library filenames that ship with symfluence.
 * Any library_file value whose basename matches one of these
 * will be rewritten to point to dist/lib/<name>.
 */
const BMI_LIBRARIES = new Set([
  'libcfebmi.dylib', 'libcfebmi.so',
  'libslothmodel.dylib', 'libslothmodel.so',
  'libsurfacebmi.dylib', 'libsurfacebmi.so',
  'libpetbmi.dylib', 'libpetbmi.so',
  'libtopmodelbmi.dylib', 'libtopmodelbmi.so',
  'liblasambmi.dylib', 'liblasambmi.so',
  'libiso_c_bmi.dylib', 'libiso_c_bmi.so',
  'libwmfire.dylib', 'libwmfire.so',
  'libsumma.dylib', 'libsumma.so',
]);

/**
 * Recursively walk a JSON object and rewrite any "library_file" value
 * whose basename is a known BMI library to point to distLibDir.
 *
 * Returns true if any value was patched.
 */
function patchLibraryPaths(obj, distLibDir) {
  if (obj === null || typeof obj !== 'object') return false;

  let patched = false;

  if (Array.isArray(obj)) {
    for (const item of obj) {
      if (patchLibraryPaths(item, distLibDir)) patched = true;
    }
    return patched;
  }

  for (const [key, value] of Object.entries(obj)) {
    if (key === 'library_file' && typeof value === 'string') {
      const basename = path.basename(value);
      if (BMI_LIBRARIES.has(basename)) {
        const resolved = path.join(distLibDir, basename);
        if (fs.existsSync(resolved)) {
          obj[key] = resolved;
          patched = true;
        }
      }
    } else if (typeof value === 'object') {
      if (patchLibraryPaths(value, distLibDir)) patched = true;
    }
  }

  return patched;
}

/**
 * For ngen: detect the realization config (last .json arg), patch
 * library_file paths, and return the path to a temp patched copy.
 * Returns null if no patching is needed.
 */
function maybePatchRealization(args, distLibDir) {
  // ngen usage: ngen <catchment> <subset> <nexus> <subset> <realization.json>
  // The realization config is the last argument and ends in .json
  if (args.length === 0) return null;

  const lastArg = args[args.length - 1];
  if (!lastArg.endsWith('.json') || !fs.existsSync(lastArg)) return null;

  let config;
  try {
    config = JSON.parse(fs.readFileSync(lastArg, 'utf-8'));
  } catch (_) {
    return null;
  }

  if (!patchLibraryPaths(config, distLibDir)) return null;

  // Write patched config to a temp file next to the original
  const dir = path.dirname(lastArg);
  const base = path.basename(lastArg, '.json');
  const patchedPath = path.join(dir, `${base}_npmpatched.json`);

  fs.writeFileSync(patchedPath, JSON.stringify(config, null, 2) + '\n');
  return patchedPath;
}

function run(toolName) {
  const distDir = path.join(__dirname, '..', 'dist');
  const distBinDir = path.join(distDir, 'bin');
  const distLibDir = path.join(distDir, 'lib');

  const binary = path.join(distBinDir, toolName);

  if (!fs.existsSync(binary)) {
    console.error(
      `${toolName}: binary not found at ${binary}\n` +
      `Run 'npm install -g symfluence' to download pre-built binaries.`
    );
    process.exit(1);
  }

  const env = { ...process.env };
  if (fs.existsSync(distBinDir)) {
    env.PATH = `${distBinDir}${path.delimiter}${env.PATH || ''}`;
  }
  if (fs.existsSync(distLibDir) && process.platform === 'linux') {
    env.LD_LIBRARY_PATH = `${distLibDir}${path.delimiter}${env.LD_LIBRARY_PATH || ''}`;
  }

  let args = process.argv.slice(2);
  let patchedFile = null;

  // Auto-patch realization config for ngen
  if (toolName === 'ngen') {
    patchedFile = maybePatchRealization(args, distLibDir);
    if (patchedFile) {
      args = args.slice(0, -1).concat(patchedFile);
      console.error(
        `[symfluence] Auto-patched BMI library paths for this system.\n` +
        `[symfluence] Using: ${patchedFile}`
      );
    }
  }

  try {
    execFileSync(binary, args, { stdio: 'inherit', env });
  } catch (err) {
    process.exit(err.status || 1);
  } finally {
    // Clean up temp patched file
    if (patchedFile) {
      try { fs.unlinkSync(patchedFile); } catch (_) {}
    }
  }
}

module.exports = { run };
