import { describe, expect, it } from "vitest";
import {
  estimateTitle,
  folderLabel,
  formatDuration,
  formatEstimate,
  LONG_EDGE_MIN,
  longEdgeRange,
  photoCount,
  snapLongEdge,
} from "./format";

describe("formatEstimate", () => {
  it("rounds to whole megabytes and never says less than 1 MB", () => {
    expect(formatEstimate(120_000_000)).toBe("≈ 120 MB");
    expect(formatEstimate(74_700_000)).toBe("≈ 75 MB");
    expect(formatEstimate(10_000)).toBe("≈ 1 MB");
    expect(formatEstimate(0)).toBe("≈ 1 MB");
  });

  it("switches to gigabytes with one decimal from 1000 MB, as the mockup does", () => {
    expect(formatEstimate(900_000_000)).toBe("≈ 900 MB");
    expect(formatEstimate(999_400_000)).toBe("≈ 999 MB");
    expect(formatEstimate(1_000_000_000)).toBe("≈ 1.0 GB");
    expect(formatEstimate(3_600_000_000)).toBe("≈ 3.6 GB");
  });
});

describe("formatDuration", () => {
  it("says seconds under a minute and a half, minutes after", () => {
    expect(formatDuration(0)).toBe("");
    expect(formatDuration(0.4)).toBe("about 1 s");
    expect(formatDuration(3.8)).toBe("about 4 s");
    expect(formatDuration(89)).toBe("about 89 s");
    expect(formatDuration(180)).toBe("about 3 min");
  });

  it("joins with the size for a tooltip", () => {
    expect(estimateTitle({ bytes: 75_000_000, seconds: 3.8, adjusted_photos: 0, dng_approximate: false })).toBe("≈ 75 MB, about 4 s");
    expect(estimateTitle({ bytes: 75_000_000, seconds: 0, adjusted_photos: 0, dng_approximate: false })).toBe("≈ 75 MB");
  });
});

describe("folderLabel and photoCount", () => {
  it("takes the last part of a path, with or without a trailing slash", () => {
    expect(folderLabel("~/Pictures/Lisbon export")).toBe("Lisbon export");
    expect(folderLabel("/Users/me/Pictures/Lisbon export/")).toBe("Lisbon export");
    expect(folderLabel("Exports")).toBe("Exports");
  });

  it("counts photos", () => {
    expect(photoCount(1)).toBe("1 photo");
    expect(photoCount(3)).toBe("3 photos");
    expect(photoCount(0)).toBe("0 photos");
  });
});

describe("long edge", () => {
  it("runs from 1000 px to the full size in 20 px steps", () => {
    expect(longEdgeRange(8640)).toEqual({ min: 1000, max: 8640 });
    expect(longEdgeRange(8645)).toEqual({ min: 1000, max: 8640 });
    expect(longEdgeRange(6000)).toEqual({ min: 1000, max: 6000 });
  });

  it("is not offered for a photo that is hardly bigger than the least", () => {
    expect(longEdgeRange(1000)).toBeNull();
    expect(longEdgeRange(1019)).toBeNull();
    expect(longEdgeRange(1020)).toEqual({ min: 1000, max: 1020 });
  });

  it("snaps a value onto the grid and into the range", () => {
    expect(snapLongEdge(3000, 8640)).toBe(3000);
    expect(snapLongEdge(3009, 8640)).toBe(3000);
    expect(snapLongEdge(3011, 8640)).toBe(3020);
    expect(snapLongEdge(500, 8640)).toBe(LONG_EDGE_MIN);
    expect(snapLongEdge(12000, 8640)).toBe(8640);
    expect(snapLongEdge(3000, 800)).toBe(800);
  });
});
