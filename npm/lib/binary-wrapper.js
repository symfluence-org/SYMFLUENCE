/**
 * Generic CLI wrapper for symfluence bundled binaries.
 *
 * Each tool shim (bin/ngen, bin/summa, etc.) requires this module
 * and it resolves the native binary from dist/bin/<toolName>.
 */

const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

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

  try {
    execFileSync(binary, process.argv.slice(2), { stdio: 'inherit', env });
  } catch (err) {
    process.exit(err.status || 1);
  }
}

module.exports = { run };
