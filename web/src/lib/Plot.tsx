// Wraps plotly.js-dist-min via the react-plotly.js factory so we don't pull in
// the full ~5 MB plotly.js bundle. This file is the single import point for
// Plotly across the app.
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);
export default Plot;
export { Plotly };
