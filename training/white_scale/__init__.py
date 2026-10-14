"""Per-body white scale of the camera registry (docs/03, "White scale of stand-in bodies").

A body without published curves renders with a sibling's or the A7R IV's profile, and the profile's light
locus then reads its whites wrongly. ``fit`` finds the two numbers per body that put the body's raw white
on the profile's scale; ``sr2`` reads the camera's own white-balance presets out of an ARW file.
"""
