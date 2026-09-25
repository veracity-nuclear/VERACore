# Changelog

All notable changes to VERACore are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-31

This version is a substantial rework of the original VERACore by Kitware Inc., which was
released under the Apache License 2.0. See `CHANGELOG-kitware.md` for the
history of the original project and `NOTICE` for attribution.

VERACore is a desktop application for visualizing and interpreting output data
from VERA (Virtual Environment for Reactor Applications) codes. It reads VERA
output (`.out`) HDF5 files and presents reactor core data through multiple
linked, interactive views arranged in a configurable grid.

### Capabilities in this release

- Loads VERA output HDF5 files and automatically categorizes datasets by type
  (pin, assembly, axial, nodal, radial, channel, detector, and computational
  core variants).
- Multiple simultaneous views in a draggable, resizable grid, including core,
  assembly, axial, volume, tabular, and plot views. Each view can be locked to
  hold its selection independently of the global core position.
- Steps through state points (e.g. exposure/burnup), with all unlocked views
  updating to the selected state.
- Handles quarter-core symmetry, masking reflected regions where appropriate.
- Derived datasets computed from a source array using average, standard
  deviation, or root-mean-square reductions over assembly, axial, radial,
  nodal, or core axes.
- Dataset comparison (diff) between two sources, including interpolation across
  differing axial meshes.
- Value thresholding to isolate data within a range of interest.
- Session save and restore of the layout, view selections, and loaded files.
- In-application version notifications when a newer release is available.

[2.0.0]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.0

## [2.0.1] - 2026-08-06

- This version adds functionality for viewing numeric values of nodal and assembly valued datasets on the core view, axial view, and surface core view.

[2.0.1]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.1

## [2.0.2] - 2026-08-07

- This version adds functionality for changing the decimals displayed on the Surface Assembly View.

[2.0.2]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.2

## [2.0.3] - 2026-08-22

- This version fixes bugs with difference dataset creation and bugs with saving sessions with locked views.

[2.0.3]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.3

## [2.0.4] - 2026-08-26

- This version fixes bugs with difference dataset creation and bugs with saving sessions with locked views.

[2.0.4]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.4

## [2.0.5] - 2026-09-11

- This version adds vera streams from roms

[2.0.5]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.5

## [2.0.6] - 2026-09-16

- This version now identifies datasets with up to 51 energy groups, now determines if the core_map is unused and if the so, uses the computational_core_map as the core_map if present.

[2.0.6]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.6

## [2.0.7] - 2026-09-19

- This version now identifies surface and energy datasets belonging to the normal core_map and axial_mesh
instead of just for computational_core_map and comp_axial_mesh.

[2.0.7]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.7

## [2.0.8] - 2026-09-25

This version adds the following:
 - A UI dropdown to allow users change color map theme from UI.
 - Individual groups from an energy dataset can now be selected from the dataset picker, and viewed on their own in their respective views.
 - A button to toggle between the color bar range being derived over all states or just the current selected state.

[2.0.8]: https://github.com/veracity-nuclear/VERACore/releases/tag/v2.0.8
