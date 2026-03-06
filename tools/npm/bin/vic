#!/usr/bin/env node
/**
 * Shared tool wrapper - delegates to the corresponding binary in dist/bin/.
 * Each tool wrapper (summa, mizuroute, etc.) requires this module.
 * The tool name is inferred from the script filename in process.argv[1].
 */

const path = require('path');
const { execFileSync } = require('child_process');
const fs = require('fs');

const tool = path.basename(process.argv[1]);
const bin = path.join(__dirname, '..', 'dist', 'bin', tool);

if (!fs.existsSync(bin)) {
  console.error(`Error: ${tool} is not installed.`);
  console.error(`Run 'npm install -g symfluence' to download binaries.`);
  process.exit(1);
}

try {
  execFileSync(bin, process.argv.slice(2), { stdio: 'inherit' });
} catch (e) {
  process.exit(e.status || 1);
}
