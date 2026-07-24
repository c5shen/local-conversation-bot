// Capture worklet: forwards mono float32 frames from the mic to the main thread.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // Copy the frame (the underlying buffer is reused by the engine).
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
