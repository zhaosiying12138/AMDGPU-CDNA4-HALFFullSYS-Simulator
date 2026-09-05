param(
    [Parameter(Mandatory = $true)]
    [string]$Title,

    [Parameter(Mandatory = $true)]
    [string]$LinuxCommand,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$MetadataPath = "",

    [string]$Distro = "Ubuntu",
    [Parameter(Mandatory = $true)]
    [string]$Workspace,
    [int]$TimeoutSeconds = 7200,
    [int]$HoldSeconds = 20
)

# Capture a real Windows Terminal window that shows, on the Ubuntu (purple)
# profile, a nested PowerShell prompt running `wsl -d Ubuntu`, the genuine
# colored Ubuntu prompt, and the live output of -LinuxCommand.
# The command itself always runs for real; only the two prompt lines are
# rendered so the single captured frame explains how the shell was entered.
# Adapted from the pypto-love-tensor-ir capture tooling (same DPI/maximize/
# PrintWindow lessons; see this repo's blog for the rationale).

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class AmdgpuSimWindowCapture {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr extraData);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint flags);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int command);

    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr hWnd, int x, int y, int width, int height, bool repaint);

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
"@

# Without DPI awareness GetWindowRect/PrintWindow operate on virtualized
# coordinates and the capture only covers the top-left logical quadrant of
# the window on scaled displays; make this process DPI-aware first.
[void][AmdgpuSimWindowCapture]::SetProcessDPIAware()

function Find-WindowByExactTitle([string]$ExpectedTitle) {
    $script:MatchedWindow = [IntPtr]::Zero
    $callback = [AmdgpuSimWindowCapture+EnumWindowsProc]{
        param([IntPtr]$Handle, [IntPtr]$Unused)
        if (-not [AmdgpuSimWindowCapture]::IsWindowVisible($Handle)) {
            return $true
        }
        $text = New-Object System.Text.StringBuilder 1024
        [void][AmdgpuSimWindowCapture]::GetWindowText($Handle, $text, $text.Capacity)
        if ($text.ToString() -eq $ExpectedTitle) {
            $script:MatchedWindow = $Handle
            return $false
        }
        return $true
    }
    [void][AmdgpuSimWindowCapture]::EnumWindows($callback, [IntPtr]::Zero)
    return $script:MatchedWindow
}

if ($TimeoutSeconds -le 0 -or $HoldSeconds -lt 10) {
    throw "TimeoutSeconds must be positive and HoldSeconds must be at least 10"
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    throw "OutputPath must be an absolute Windows path"
}
if ($MetadataPath -and -not [System.IO.Path]::IsPathRooted($MetadataPath)) {
    throw "MetadataPath must be an absolute Windows path"
}
$windowsHome = $env:USERPROFILE
if (-not $windowsHome) {
    throw "USERPROFILE is not set; cannot render the PowerShell prompt line"
}

$captureStartedUtc = [DateTime]::UtcNow.ToString("o")
$uniqueTitle = "amdgpu-sim - $Title - $([Guid]::NewGuid().ToString('N').Substring(0, 8))"
$nonce = [Guid]::NewGuid().ToString('N')
$marker = "/tmp/asim-ps-capture-$nonce.done"
$inner = "/tmp/asim-ps-capture-$nonce-inner.sh"
$outer = "/tmp/asim-ps-capture-$nonce-outer.sh"
$uncRoot = "\\wsl.localhost\$Distro"
$markerWindows = "$uncRoot\tmp\asim-ps-capture-$nonce.done"
$innerWindows = "$uncRoot\tmp\asim-ps-capture-$nonce-inner.sh"
$outerWindows = "$uncRoot\tmp\asim-ps-capture-$nonce-outer.sh"

# The command text and workspace travel base64-encoded so no quoting in the
# rendered shell lines can break the wrapper.
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($LinuxCommand))
$workspaceEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Workspace))
$distroEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Distro))
$titleQuoted = $uniqueTitle.Replace("'", "")

# Inner script: runs inside the nested `wsl -d Ubuntu` shell. It prints a
# genuine-format colored Ubuntu prompt (real user/host/cwd), echoes the exact
# command line, runs it for real, then reports the exit code and flags done.
$innerBody = @"
set -o pipefail
workspace="`$(printf '%s' '$workspaceEncoded' | base64 -d)"
cd "`$workspace"
command_text="`$(printf '%s' '$encoded' | base64 -d)"
short_pwd="`${PWD/#`$HOME/\~}"
printf '\e[01;32m%s@%s\e[00m:\e[01;34m%s\e[00m%s' "`$(id -un)" "`$(hostname)" "`$short_pwd" '`$ '
printf '%s\n' "`$command_text"
eval "`$command_text"
status=`$?
printf '\nexit_code=%s finished_utc=%s\n' "`$status" "`$(date -u +%FT%TZ)"
printf '%s\n' "`$status" > '$marker'
exit `$status
"@

# Outer script: runs in the Windows Terminal Ubuntu tab. It sets the window
# title, then launches a nested powershell.exe whose prompt visibly enters
# `wsl -d Ubuntu`, and holds the window open after the command finishes.
$outerBody = @"
set -o pipefail
printf '\033]0;$titleQuoted\007'
distro="`$(printf '%s' '$distroEncoded' | base64 -d)"
workspace="`$(printf '%s' '$workspaceEncoded' | base64 -d)"
powershell.exe -NoLogo -NoProfile -Command "Set-Location '$windowsHome'; Write-Host -NoNewline ('PS ' + (Get-Location).Path + '> '); Write-Host ('wsl -d ' + '`$distro' + ' --cd ' + '`$workspace'); wsl.exe -d '`$distro' --cd '`$workspace' -- bash '$inner'; exit `\`$LASTEXITCODE"
status=`$?
sleep $HoldSeconds
exit `$status
"@

$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($innerWindows, $innerBody, $utf8)
[System.IO.File]::WriteAllText($outerWindows, $outerBody, $utf8)

$safeDistro = $Distro.Replace('"', '')
$safeTitle = $uniqueTitle.Replace('"', '')
$argumentLine = "-w new nt -p `"$safeDistro`" --title `"$safeTitle`" " +
    "wsl.exe -d `"$safeDistro`" -- bash --noprofile --norc $outer"
Start-Process -FilePath "wt.exe" -ArgumentList $argumentLine | Out-Null

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
$window = [IntPtr]::Zero
do {
    Start-Sleep -Milliseconds 250
    $window = Find-WindowByExactTitle $uniqueTitle
    $finished = Test-Path -LiteralPath $markerWindows
    if ([DateTime]::UtcNow -ge $deadline) {
        throw "Timed out waiting for terminal evidence command"
    }
} while ($window -eq [IntPtr]::Zero -or -not $finished)

[void][AmdgpuSimWindowCapture]::ShowWindow($window, 3)
[void][AmdgpuSimWindowCapture]::SetForegroundWindow($window)
Start-Sleep -Milliseconds 400
# Force the window onto the full primary working area in physical pixels so
# the whole command and its output are visible regardless of the terminal's
# default size; maximize alone does not reliably resize Windows Terminal.
$work = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
[void][AmdgpuSimWindowCapture]::MoveWindow(
    $window, $work.X, $work.Y, $work.Width, $work.Height, $true
)
Start-Sleep -Milliseconds 1200

$rect = New-Object AmdgpuSimWindowCapture+RECT
if (-not [AmdgpuSimWindowCapture]::GetWindowRect($window, [ref]$rect)) {
    throw "GetWindowRect failed"
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -lt 320 -or $height -lt 200) {
    throw "Terminal window is unexpectedly small: ${width}x${height}"
}

$directory = [System.IO.Path]::GetDirectoryName($OutputPath)
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$captureMethod = "PrintWindow"
$visibleSamples = 0
$sampleCount = 0
try {
    $hdc = $graphics.GetHdc()
    try {
        $printed = [AmdgpuSimWindowCapture]::PrintWindow($window, $hdc, 2)
    }
    finally {
        $graphics.ReleaseHdc($hdc)
    }
    if (-not $printed) {
        $captureMethod = "CopyFromScreen"
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
    }
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)

    $stepX = [Math]::Max(1, [int]($width / 96))
    $stepY = [Math]::Max(1, [int]($height / 54))
    for ($y = 0; $y -lt $height; $y += $stepY) {
        for ($x = 0; $x -lt $width; $x += $stepX) {
            $pixel = $bitmap.GetPixel($x, $y)
            $sampleCount += 1
            if (($pixel.R + $pixel.G + $pixel.B) -gt 24) {
                $visibleSamples += 1
            }
        }
    }
    if ($sampleCount -eq 0 -or $visibleSamples -lt 16) {
        throw "Captured terminal image is blank or unreadable: visible_samples=$visibleSamples/$sampleCount"
    }
}
finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}

$statusText = [System.IO.File]::ReadAllText($markerWindows).Trim()
Remove-Item -LiteralPath $markerWindows, $innerWindows, $outerWindows -Force -ErrorAction SilentlyContinue
[void][AmdgpuSimWindowCapture]::PostMessage($window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)

if ($statusText -ne "0") {
    throw "Evidence command failed with exit code $statusText; screenshot retained at $OutputPath"
}

$file = Get-Item -LiteralPath $OutputPath
if ($file.Length -lt 4096) {
    throw "Captured PNG is unexpectedly small: $($file.Length) bytes"
}
if ($MetadataPath) {
    $metadataDirectory = [System.IO.Path]::GetDirectoryName($MetadataPath)
    [System.IO.Directory]::CreateDirectory($metadataDirectory) | Out-Null
    $metadata = [ordered]@{
        schema = 1
        kind = "windows-terminal-capture"
        status = "pass"
        role = $Title
        command = $LinuxCommand
        workspace = $Workspace
        distro = $Distro
        host = "Windows Terminal"
        theme = "Ubuntu profile, nested PowerShell prompt -> wsl -d Ubuntu"
        unique_window_title = $uniqueTitle
        started_utc = $captureStartedUtc
        finished_utc = [DateTime]::UtcNow.ToString("o")
        window_width = $width
        window_height = $height
        capture_method = $captureMethod
        visible_samples = $visibleSamples
        sample_count = $sampleCount
        output_path = $OutputPath
        output_bytes = $file.Length
        output_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputPath).Hash.ToLowerInvariant()
        exit_code = [int]$statusText
    }
    $metadataJson = $metadata | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($MetadataPath, $metadataJson + "`n", $utf8)
}
Write-Output $file.FullName
