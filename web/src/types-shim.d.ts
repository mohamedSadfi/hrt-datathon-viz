// Type shims for libraries that ship JS-only or have incomplete types.

declare module 'plotly.js-dist-min' {
  // We re-export the full plotly.js types from this min bundle.
  // Behaviorally identical at runtime.
  import * as Plotly from 'plotly.js';
  export = Plotly;
}

declare module 'react-plotly.js/factory' {
  import type { ComponentType } from 'react';
  import type { PlotParams } from 'react-plotly.js';
  function createPlotlyComponent(plotly: unknown): ComponentType<PlotParams>;
  export default createPlotlyComponent;
}
