import katex from 'katex';
import { useMemo } from 'react';

export default function MathBlock({
  formula,
  display = false,
}: {
  formula: string;
  display?: boolean;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(formula, {
        displayMode: display,
        throwOnError: false,
        output: 'html',
      }),
    [formula, display],
  );
  return (
    <span
      className={display ? 'block my-2' : 'inline-block'}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
