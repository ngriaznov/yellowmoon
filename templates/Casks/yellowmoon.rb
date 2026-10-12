cask "yellowmoon" do
  version "0.1.0"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

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
  binary "#{appdir}/Yellowmoon.app/Contents/MacOS/yellowmoon-cli", target: "yellowmoon"

  zap trash: [
    "~/Library/Application Support/app.yellowmoon",
    "~/Library/Caches/app.yellowmoon",
    "~/Library/Preferences/app.yellowmoon.plist",
    "~/Library/Saved Application State/app.yellowmoon.savedState",
    "~/Library/WebKit/app.yellowmoon",
  ]

  # Remove this caveat if Apple Developer ID signing and notarisation are adopted (docs/16).
  caveats <<~EOS
    Yellowmoon is not code-signed yet. If macOS refuses to open it, either install
    with --no-quarantine or run:
      xattr -cr /Applications/Yellowmoon.app
  EOS
end
