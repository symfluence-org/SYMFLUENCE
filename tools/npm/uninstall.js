#!/usr/bin/env node
/**
 * SYMFLUENCE npm preuninstall hook
 * Removes tool shims from npm's global bin directory that npm may not clean up.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const TOOLS = [
  'symfluence', 'summa', 'mizuroute', 'ngen', 'fuse', 'taudem',
  'vic', 'swat', 'prms', 'mhm', 'crhm', 'wrfhydro', 'watflood',
  'mf6', 'parflow', 'pihm', 'gsflow', 'clm', 'rhessys', 'hype',
  'mesh', 'wmfire'
];

function getNpmBinDir() {
  try {
    return execSync('npm prefix -g', { encoding: 'utf8' }).trim() + '/bin';
  } catch {
    return null;
  }
}

function main() {
  const binDir = getNpmBinDir();
  if (!binDir) {
    return;
  }

  const pkgDir = path.resolve(__dirname);
  let removed = 0;

  for (const tool of TOOLS) {
    const shimPath = path.join(binDir, tool);
    try {
      const target = fs.readlinkSync(shimPath);
      // Only remove if it points into this package
      const resolved = path.resolve(path.dirname(shimPath), target);
      if (resolved.startsWith(pkgDir + path.sep) || resolved.startsWith(pkgDir + '/')) {
        fs.unlinkSync(shimPath);
        removed++;
      }
    } catch {
      // Not a symlink or doesn't exist — skip
    }
  }

  if (removed > 0) {
    console.log(`Removed ${removed} symfluence shim(s) from ${binDir}`);
  }
}

main();
