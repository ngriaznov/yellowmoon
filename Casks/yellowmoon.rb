cask "yellowmoon" do
  version "0.1.1"
  sha256 "911a7f1ff7ad75d9043aa4f8dc56a3b1f846fb730c69e305b0fa6a3fb536aa4c"

  url "https://github.com/ngriaznov/yellowmoon/releases/download/v#{version}/Yellowmoon_#{version}_universal.dmg"
  name "Yellowmoon"
  desc "Cinema colour for Sony raw files"
  homepage "https://github.com/ngriaznov/yellowmoon"

  livecheck do
    url :url
    strategy :github_latest
  end

  auto_updates true
  depends_on macos: ">= :monterey"

  app "Yellowmoon.app"
  binary "#{appdir}/Yellowmoon.app/Contents/MacOS/ym"

  zap trash: [
    "~/Library/Application Support/app.yellowmoon",
    "~/Library/Caches/app.yellowmoon",
    "~/Library/Preferences/app.yellowmoon.plist",
    "~/Library/Saved Application State/app.yellowmoon.savedState",
    "~/Library/WebKit/app.yellowmoon",
  ]

  # Yellowmoon ships without an Apple Developer ID (Mykyta, 2026-10-15; docs/16, README "Allow Yellowmoon to open").
  caveats <<~EOS
    Yellowmoon is not signed with an Apple Developer ID. If macOS refuses to open it,
    run once:
      xattr -dr com.apple.quarantine /Applications/Yellowmoon.app
    or allow it in System Settings > Privacy & Security > Open Anyway.
  EOS
end
