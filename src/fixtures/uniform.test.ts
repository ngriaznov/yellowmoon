/** The mock's picture of a graded Colour Uniform node (`docs/32` §12a.1): the pull, then the corrector's trim and grade on its result. */
import { describe, expect, it } from "vitest";
import { neutralGrade, newColorUniform } from "../state/settingsGraph";
import { adjustForNode } from "./grade";
import { SCENES } from "./scenes";
import { uniformGradedScene, uniformScene } from "./uniform";

const scene = SCENES[2]!;
const live = { ...newColorUniform(2, [1, 0]), target: { hue_deg: 26, saturation: 9.7, lightness_ev: 0 }, compress: { hue: 80, saturation: 80, lightness: 40 } };

describe("uniformGradedScene", () => {
  it("is the pull alone for a node with no trim and no grade", () => {
    expect(uniformGradedScene(scene, live)).toEqual(uniformScene(scene, live.target, live.compress));
  });

  it("runs the grade on the result of the pull, not the pull on the graded scene", () => {
    const graded = { ...live, trim: { exposure_ev: 0.8, temperature: 20, tint: 0 }, grade: { ...neutralGrade(), saturation: { kind: "manual" as const, value: -60 } } };
    const look = { bypass: false, trim: graded.trim, grade: graded.grade };
    const pullThenGrade = adjustForNode(uniformScene(scene, live.target, live.compress), look);
    const gradeThenPull = uniformScene(adjustForNode(scene, look), live.target, live.compress);
    expect(uniformGradedScene(scene, graded)).toEqual(pullThenGrade);
    expect(pullThenGrade).not.toEqual(gradeThenPull);
    expect(pullThenGrade).not.toEqual(uniformScene(scene, live.target, live.compress));
  });
});
