#!/usr/bin/env node
/**
 * SYMFLUENCE npm installer
 * Downloads and extracts pre-built binaries from GitHub Releases
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');
const { execSync } = require('child_process');
const { getPlatform, getPlatformName } = require('./lib/platform');

const PACKAGE_VERSION = require('./package.json').version;
const GITHUB_REPO = 'symfluence-org/SYMFLUENCE';

/**
 * Construct the download URL for the current platform
 * @param {string} platform - Platform identifier (e.g., 'macos-arm64')
 * @returns {string} Full download URL
 */
function getDownloadUrl(platform) {
  const tag = `v${PACKAGE_VERSION}`;
  const filename = `symfluence-tools-${tag}-${platform}.tar.gz`;
  return `https://github.com/${GITHUB_REPO}/releases/download/${tag}/${filename}`;
}

/**
 * Download a file from URL with progress tracking
 * @param {string} url - URL to download from
 * @param {string} dest - Destination file path
 * @returns {Promise<void>}
 */
async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const request = https.get(url, {
      headers: { 'User-Agent': 'symfluence-npm-installer' }
    }, (response) => {
      // Handle redirects
      if (response.statusCode === 302 || response.statusCode === 301) {
        file.close();
        fs.unlinkSync(dest);
        downloadFile(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        reject(new Error(
          `Download failed: ${response.statusCode} ${response.statusMessage}\n` +
          `URL: ${url}`
        ));
        return;
      }

      const totalBytes = parseInt(response.headers['content-length'], 10);
      let downloadedBytes = 0;
      let lastPercent = -1;

      response.on('data', (chunk) => {
        downloadedBytes += chunk.length;
        const percent = Math.floor((downloadedBytes / totalBytes) * 100);

        // Only update display every 5% to reduce output noise
        if (percent !== lastPercent && percent % 5 === 0) {
          const mb = (downloadedBytes / 1024 / 1024).toFixed(1);
          const totalMb = (totalBytes / 1024 / 1024).toFixed(1);
          process.stdout.write(`\r📥 Downloading... ${percent}% (${mb}/${totalMb} MB)`);
          lastPercent = percent;
        }
      });

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log('\n✅ Download complete');
        resolve();
      });

      file.on('error', (err) => {
        fs.unlinkSync(dest);
        reject(err);
      });
    });

    request.on('error', (err) => {
      if (fs.existsSync(dest)) {
        fs.unlinkSync(dest);
      }
      reject(err);
    });
  });
}

/**
 * Verify file checksum against published SHA256
 * @param {string} file - Path to file to verify
 * @param {string} checksumUrl - URL of .sha256 file
 * @returns {Promise<void>}
 */
async function verifyChecksum(file, checksumUrl) {
  console.log('🔐 Verifying checksum...');

  try {
    // Download checksum file
    const checksumData = await new Promise((resolve, reject) => {
      let data = '';
      https.get(checksumUrl, {
        headers: { 'User-Agent': 'symfluence-npm-installer' }
      }, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          // Follow redirect
          https.get(res.headers.location, (redirectRes) => {
            redirectRes.on('data', chunk => data += chunk);
            redirectRes.on('end', () => resolve(data));
          }).on('error', reject);
          return;
        }
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      }).on('error', reject);
    });

    // Extract expected hash (format: "hash  filename")
    const expectedHash = checksumData.trim().split(/\s+/)[0];

    // Calculate actual hash
    const fileBuffer = fs.readFileSync(file);
    const hash = crypto.createHash('sha256');
    hash.update(fileBuffer);
    const actualHash = hash.digest('hex');

    if (expectedHash.toLowerCase() !== actualHash.toLowerCase()) {
      throw new Error(
        'Checksum mismatch! File may be corrupted.\n' +
        `  Expected: ${expectedHash}\n` +
        `  Actual:   ${actualHash}`
      );
    }

    console.log('✅ Checksum verified');
  } catch (err) {
    console.warn('⚠️  Could not verify checksum:', err.message);
    console.warn('   Proceeding anyway, but installation may be corrupted...');
  }
}

/**
 * Extract tarball to destination directory
 * @param {string} tarball - Path to tarball
 * @param {string} destDir - Destination directory
 */
function extractTarball(tarball, destDir) {
  console.log('📦 Extracting binaries...');

  // Use --strip-components=1 to remove the top-level directory
  const extractCmd = `tar -xzf "${tarball}" -C "${destDir}" --strip-components=1`;

  try {
    execSync(extractCmd, { stdio: 'inherit' });
    console.log('✅ Extraction complete');
  } catch (err) {
    throw new Error(`Extraction failed: ${err.message}`);
  }
}

/**
 * Main installation function
 */
async function install() {
  console.log('╔════════════════════════════════════════════╗');
  console.log('║   SYMFLUENCE Binary Installer              ║');
  console.log('╚════════════════════════════════════════════╝\n');

  // Detect platform
  let platform;
  try {
    platform = getPlatform();
  } catch (err) {
    console.error('❌', err.message);
    console.error('\n📖 For manual installation, see:');
    console.error('   https://github.com/symfluence-org/SYMFLUENCE#installation\n');
    process.exit(1);
  }

  console.log(`📍 Platform: ${getPlatformName()} (${platform})`);
  console.log(`📦 Version: ${PACKAGE_VERSION}\n`);

  const url = getDownloadUrl(platform);
  const checksumUrl = `${url}.sha256`;

  console.log(`🔗 Downloading from GitHub Releases...`);
  console.log(`   ${url}\n`);

  const distDir = path.join(__dirname, 'dist');
  const tarballPath = path.join(__dirname, 'symfluence-tools.tar.gz');

  // Clean and create dist directory
  if (fs.existsSync(distDir)) {
    console.log('🧹 Cleaning previous installation...');
    fs.rmSync(distDir, { recursive: true, force: true });
  }
  fs.mkdirSync(distDir, { recursive: true });

  try {
    // Download tarball
    await downloadFile(url, tarballPath);

    // Verify checksum
    await verifyChecksum(tarballPath, checksumUrl);

    // Extract
    extractTarball(tarballPath, distDir);

    // Cleanup tarball
    fs.unlinkSync(tarballPath);

    // Display installation info
    console.log('\n╔════════════════════════════════════════════╗');
    console.log('║   🎉 Installation Complete!                ║');
    console.log('╚════════════════════════════════════════════╝\n');

    console.log('📦 Installed Tools:');
    const binDir = path.join(distDir, 'bin');
    if (fs.existsSync(binDir)) {
      const tools = fs.readdirSync(binDir).filter(f => {
        const fullPath = path.join(binDir, f);
        return fs.statSync(fullPath).isFile();
      });
      tools.forEach(tool => console.log(`   ✓ ${tool}`));
    }

    console.log('\n📖 Next Steps:');
    console.log('   1. Check installation: symfluence --help');
    console.log('   2. View available tools: ls $(npm root -g)/symfluence/dist/bin');
    console.log('   3. Install Python package: pip install symfluence\n');

    console.log('📚 Documentation: https://github.com/symfluence-org/SYMFLUENCE\n');

  } catch (err) {
    console.error('\n❌ Installation failed:', err.message);
    console.error('\n📖 Troubleshooting:');
    console.error('   1. Check your internet connection');
    console.error('   2. Verify the release exists:');
    console.error(`      https://github.com/${GITHUB_REPO}/releases/tag/v${PACKAGE_VERSION}`);
    console.error('   3. Check system requirements:');
    console.error('      https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md');
    console.error('   4. Try manual installation:');
    console.error('      https://github.com/symfluence-org/SYMFLUENCE#installation\n');

    // Clean up on failure
    if (fs.existsSync(tarballPath)) {
      fs.unlinkSync(tarballPath);
    }
    if (fs.existsSync(distDir)) {
      fs.rmSync(distDir, { recursive: true, force: true });
    }

    process.exit(1);
  }
}

// Run installer if executed directly (not required)
if (require.main === module) {
  install();
}
