import { app } from "../../scripts/app.js";

const NODE_CLASS = "bg_remove_compose";
const FIT_WIDGET = "fit_to_canvas";
const SCALE_WIDGET = "original_image_scale";

function syncScaleDisabled(node) {
  const fit = node.widgets?.find((w) => w.name === FIT_WIDGET);
  const scale = node.widgets?.find((w) => w.name === SCALE_WIDGET);
  if (!fit || !scale) return;
  scale.disabled = !!fit.value;
}

app.registerExtension({
  name: "whisker.bg_remove_compose",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;

    const fit = node.widgets?.find((w) => w.name === FIT_WIDGET);
    if (fit) {
      const origCallback = fit.callback;
      fit.callback = function (...args) {
        const result = origCallback?.apply(this, args);
        syncScaleDisabled(node);
        return result;
      };
    }

    // Workflow loads set widget values after creation.
    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = origConfigure?.apply(this, args);
      syncScaleDisabled(node);
      return result;
    };

    syncScaleDisabled(node);
  },
});
